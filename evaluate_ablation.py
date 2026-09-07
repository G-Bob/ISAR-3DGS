from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from radar_gs.data import ISARView, load_dataset
from radar_gs.io import export_ply, load_scene, save_preview, write_json
from radar_gs.losses import compress_amplitude, inverse_compress_amplitude, ssim
from radar_gs.ply_evaluation import (
    PlyTable,
    filter_by_support,
    ground_truth_metrics,
    intrinsic_point_metrics,
    load_reference_geometry,
    make_reference_voxel_grid,
    read_ascii_ply,
    save_gaussian_support_preview,
    voxel_overlap_metrics,
    voxelize_gaussian_support,
    voxelize_reference_geometry,
    write_ascii_ply,
    write_voxel_surface_ply,
)
from radar_gs.renderer import RadarVisibilityRenderer
from radar_gs.presets import load_encounter_batch_preset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate held-out ISAR views and q-thresholded Gaussian PLY files"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--out_dir")
    parser.add_argument("--selected_views")
    parser.add_argument(
        "--evaluation_views_file",
        help="Shared JSON list of exact evaluation frames; created when absent",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=["heldout", "all"], default="heldout")
    parser.add_argument("--eval_encounter_count", type=int)
    parser.add_argument(
        "--evaluation_preset_json",
        help=(
            "Evaluate exactly the encounters listed by a GEIST preset. Incomplete "
            "test manifests are accepted here, but never by the training entry point."
        ),
    )
    parser.add_argument("--max_eval_views", type=int, default=200)
    parser.add_argument("--eval_seed", type=int, default=12345)
    parser.add_argument("--preview_count", type=int, default=8)
    parser.add_argument("--background", choices=["border", "zero"], default="border")
    parser.add_argument("--q_thresholds", nargs="+", type=float, default=[0.1, 0.3, 0.5])
    parser.add_argument("--graph_neighbors", type=int, default=8)
    parser.add_argument("--connectivity_radius", type=float, default=0.0)
    parser.add_argument("--outlier_multiplier", type=float, default=3.0)
    parser.add_argument("--gt_geometry")
    parser.add_argument("--gt_sample_count", type=int, default=50000)
    parser.add_argument("--prediction_unit_scale", type=float, default=1.0)
    parser.add_argument("--reference_unit_scale", type=float, default=1.0)
    parser.add_argument("--voxel_resolution", type=int, default=256)
    parser.add_argument("--voxel_padding_fraction", type=float, default=0.05)
    parser.add_argument("--gaussian_sigma_extent", type=float, default=2.0)
    parser.add_argument("--occupancy_probability_threshold", type=float, default=0.5)
    parser.add_argument("--overlap_contribution_threshold", type=float, default=0.05)
    parser.add_argument("--no_fill_prediction_voxels", action="store_true")
    parser.add_argument("--no_export_fused_ply", action="store_true")
    parser.add_argument(
        "--distance_thresholds_m", nargs="+", type=float, default=[0.05, 0.10, 0.20]
    )
    return parser


def _training_view_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        raise FileNotFoundError(f"Selected-view report not found: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    keys: set[tuple[str, int]] = set()
    for encounter, information in report["selection"].items():
        keys.update((encounter, int(index)) for index in information["indices"])
    return keys


def _balanced_subset(views: list[ISARView], maximum: int, seed: int) -> list[ISARView]:
    if maximum <= 0 or len(views) <= maximum:
        return views
    groups: dict[str, list[ISARView]] = defaultdict(list)
    for view in views:
        groups[view.encounter].append(view)
    generator = random.Random(seed)
    names = sorted(groups)
    base = maximum // len(names)
    remainder = maximum % len(names)
    chosen: list[ISARView] = []
    for index, name in enumerate(names):
        group = sorted(groups[name], key=lambda item: item.frame_index)
        quota = min(len(group), base + (1 if index < remainder else 0))
        if quota == len(group):
            chosen.extend(group)
        elif quota > 0:
            chosen.extend(generator.sample(group, quota))
    if len(chosen) < maximum:
        chosen_keys = {(view.encounter, view.frame_index) for view in chosen}
        remaining = [
            view
            for view in views
            if (view.encounter, view.frame_index) not in chosen_keys
        ]
        chosen.extend(generator.sample(remaining, min(maximum - len(chosen), len(remaining))))
    return sorted(chosen, key=lambda item: (item.encounter, item.frame_index))[:maximum]


def _fixed_evaluation_views(
    candidates: list[ISARView],
    path: Path | None,
    maximum: int,
    seed: int,
) -> list[ISARView]:
    by_key = {(view.encounter, view.frame_index): view for view in candidates}
    if path is not None and path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
        requested = [
            (str(item["encounter"]), int(item["frame_index"]))
            for item in report["views"]
        ]
        missing = [key for key in requested if key not in by_key]
        if missing:
            raise ValueError(
                f"Shared evaluation split contains {len(missing)} unavailable frames; "
                f"first missing key is {missing[0]}"
            )
        return [by_key[key] for key in requested]

    selected = _balanced_subset(candidates, maximum, seed)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "seed": seed,
                    "maximum_views": maximum,
                    "views": [
                        {"encounter": view.encounter, "frame_index": view.frame_index}
                        for view in selected
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return selected


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _evaluate_images(
    scene,
    checkpoint: dict,
    views: list[ISARView],
    device: torch.device,
    output_dir: Path,
    preview_count: int,
    background_mode: str,
) -> dict:
    config = checkpoint["arguments"]
    renderer = RadarVisibilityRenderer(
        config["image_size"],
        config["image_size"],
        config["depth_bins"],
        config["kernel_radius"],
        config["minimum_pixel_variance"],
        config["opacity_gain"],
        config.get("visibility_pixel_size", 0.0),
        config.get("support_composition", "noisy_or"),
    ).to(device)
    gamma = float(config.get("render_gamma", 1.0))
    log_gain = float(config.get("log_amplitude_gain", 20.0))
    prediction_mode = "raw" if config.get("disable_visibility", False) else "visibility_aware"
    preview_indices = set()
    if preview_count > 0 and views:
        preview_indices = set(
            np.linspace(0, len(views) - 1, min(preview_count, len(views))).round().astype(int)
        )

    rows: list[dict[str, float | int | str]] = []
    scene.eval()
    with torch.no_grad():
        for index, view in enumerate(views):
            result = renderer(
                scene,
                view.kr,
                view.kd,
                view.dr,
                view.dd,
                support_temperature=config.get("final_support_temperature", 1.0),
            )
            signal = result.raw_image if prediction_mode == "raw" else result.image
            if background_mode == "border":
                level = torch.tensor(view.background_init, device=device)
                background = inverse_compress_amplitude(level, gamma)
            else:
                background = 0.0
            prediction = compress_amplitude(signal + background, gamma)
            raw_prediction = compress_amplitude(result.raw_image + background, gamma)
            target = view.image.to(device)
            residual = prediction - target
            mse = float(residual.square().mean())
            mae = float(residual.abs().mean())
            psnr = float(-10.0 * math.log10(max(mse, 1.0e-12)))
            ssim_value = float(ssim(prediction, target))
            log_prediction = torch.log1p(log_gain * prediction)
            log_target = torch.log1p(log_gain * target)
            log_mae = float((log_prediction - log_target).abs().mean())
            rows.append(
                {
                    "encounter": view.encounter,
                    "frame_index": view.frame_index,
                    "mae": mae,
                    "mse": mse,
                    "psnr_db": psnr,
                    "ssim": ssim_value,
                    "log_mae": log_mae,
                }
            )
            if index in preview_indices:
                save_preview(
                    output_dir / "heldout_previews" / f"{view.encounter}_frame_{view.frame_index:05d}.png",
                    target,
                    raw_prediction,
                    prediction,
                    f"{view.encounter} frame {view.frame_index}",
                    support=result.support_map,
                    overlap=result.overlap_map,
                    labels=(
                        "target | raw | raw-model prediction | soft target support | multi-Gaussian overlap"
                        if prediction_mode == "raw"
                        else "target | raw | visibility-aware | soft target support | multi-Gaussian overlap"
                    ),
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_view_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    aggregate = {
        "views": len(rows),
        "prediction_mode": prediction_mode,
        "background_mode": background_mode,
        "mean_mae": _mean([float(row["mae"]) for row in rows]),
        "mean_mse": _mean([float(row["mse"]) for row in rows]),
        "mean_psnr_db": _mean([float(row["psnr_db"]) for row in rows]),
        "mean_ssim": _mean([float(row["ssim"]) for row in rows]),
        "mean_log_mae": _mean([float(row["log_mae"]) for row in rows]),
    }
    per_encounter = {}
    for encounter in sorted({str(row["encounter"]) for row in rows}):
        subset = [row for row in rows if row["encounter"] == encounter]
        per_encounter[encounter] = {
            "views": len(subset),
            "mean_mae": _mean([float(row["mae"]) for row in subset]),
            "mean_psnr_db": _mean([float(row["psnr_db"]) for row in subset]),
            "mean_ssim": _mean([float(row["ssim"]) for row in subset]),
            "mean_log_mae": _mean([float(row["log_mae"]) for row in subset]),
        }
    aggregate["per_encounter"] = per_encounter
    return aggregate


def _evaluate_ply(
    source_ply: Path,
    output_dir: Path,
    thresholds: list[float],
    graph_neighbors: int,
    connectivity_radius: float,
    outlier_multiplier: float,
    gt_geometry: str | None,
    gt_sample_count: int,
    distance_thresholds_m: list[float],
    seed: int,
    prediction_unit_scale: float,
    reference_unit_scale: float,
    voxel_resolution: int,
    voxel_padding_fraction: float,
    gaussian_sigma_extent: float,
    fill_prediction_voxels: bool,
    occupancy_probability_threshold: float,
    overlap_contribution_threshold: float,
    export_fused_ply: bool,
) -> dict:
    table = read_ascii_ply(source_ply)
    original_count = len(table.values)
    reference_points = reference_normals = None
    voxel_spec = reference_voxels = reference_voxel_mode = None
    if gt_geometry:
        reference_points, reference_normals = load_reference_geometry(
            gt_geometry, sample_count=gt_sample_count, seed=seed
        )
        reference_points = reference_points * reference_unit_scale
        voxel_spec = make_reference_voxel_grid(
            reference_points,
            resolution=voxel_resolution,
            padding_fraction=voxel_padding_fraction,
        )
        reference_voxels, reference_voxel_mode = voxelize_reference_geometry(
            gt_geometry,
            voxel_spec,
            unit_scale=reference_unit_scale,
            fallback_points=reference_points,
        )
    elif original_count:
        all_prediction_points = np.column_stack(
            [table.column(axis) for axis in ("x", "y", "z")]
        ) * prediction_unit_scale
        voxel_spec = make_reference_voxel_grid(
            all_prediction_points,
            resolution=voxel_resolution,
            padding_fraction=voxel_padding_fraction,
        )
    results = {}
    for threshold in thresholds:
        filtered = filter_by_support(table, threshold)
        label = f"q{int(round(threshold * 100)):03d}"
        filtered_path = output_dir / "ply_thresholds" / f"gaussians_{label}.ply"
        write_ascii_ply(filtered_path, filtered)
        points = np.column_stack([filtered.column(axis) for axis in ("x", "y", "z")])
        points = points * prediction_unit_scale
        normals = None
        if all(name in filtered.properties for name in ("nx", "ny", "nz")):
            normals = np.column_stack([filtered.column(axis) for axis in ("nx", "ny", "nz")])
        support = filtered.column("target_support_probability")
        metrics = intrinsic_point_metrics(
            points,
            support,
            graph_neighbors=graph_neighbors,
            connectivity_radius=connectivity_radius,
            outlier_multiplier=outlier_multiplier,
        )
        metrics["q_threshold"] = float(threshold)
        metrics["retained_fraction"] = float(len(points) / max(original_count, 1))
        metrics["filtered_ply"] = str(filtered_path)
        if normals is None or not all(
            name in filtered.properties
            for name in ("scale_t1", "scale_t2", "scale_normal")
        ):
            raise ValueError(
                "Gaussian support visualization requires nx/ny/nz and "
                "scale_t1/scale_t2/scale_normal in the prediction PLY"
            )
        scales = np.column_stack(
            [
                filtered.column(name)
                for name in ("scale_t1", "scale_t2", "scale_normal")
            ]
        ) * prediction_unit_scale
        support_volume = voxelize_gaussian_support(
            points,
            normals,
            scales,
            support,
            voxel_spec,
            sigma_extent=gaussian_sigma_extent,
            occupancy_probability_threshold=occupancy_probability_threshold,
            overlap_contribution_threshold=overlap_contribution_threshold,
            fill_interior=fill_prediction_voxels,
        )
        preview_path = output_dir / "ply_thresholds" / f"gaussian_overlap_{label}.png"
        save_gaussian_support_preview(
            preview_path,
            support_volume,
            q_threshold=threshold,
            occupancy_probability_threshold=occupancy_probability_threshold,
        )
        fused_ply_path = None
        if export_fused_ply:
            fused_ply_path = output_dir / "ply_thresholds" / f"gaussian_fused_{label}.ply"
            write_voxel_surface_ply(fused_ply_path, support_volume.occupied_grid, voxel_spec)
        metrics["overlap_preview"] = str(preview_path)
        metrics["fused_occupancy_ply"] = (
            str(fused_ply_path) if fused_ply_path is not None else None
        )
        metrics["support_volume_summary"] = support_volume.summary()
        if reference_points is not None:
            metrics["voxel_metrics"] = voxel_overlap_metrics(
                support_volume.occupied_grid,
                reference_voxels,
                voxel_spec,
                prediction_outside_count=support_volume.outside_occupied_count,
                reference_mode=reference_voxel_mode,
                sigma_extent=gaussian_sigma_extent,
                fill_prediction_interior=fill_prediction_voxels,
                prediction_voxelization_mode="noisy_or_gaussian_support",
                occupancy_probability_threshold=occupancy_probability_threshold,
            )
            metrics["ground_truth"] = ground_truth_metrics(
                points,
                reference_points,
                distance_thresholds_m,
                prediction_normals=normals,
                reference_normals=reference_normals,
            )
        else:
            metrics["voxel_metrics"] = {
                "available": False,
                "reason": "no --gt_geometry supplied",
            }
            metrics["ground_truth"] = {
                "available": False,
                "reason": "no --gt_geometry supplied",
            }
        results[f"{threshold:g}"] = metrics
    return {"source_ply": str(source_ply), "original_point_count": original_count, "thresholds": results}


def main() -> None:
    args = build_parser().parse_args()
    if args.prediction_unit_scale <= 0 or args.reference_unit_scale <= 0:
        raise ValueError("Unit scales must be positive")
    if any(not 0.0 <= q <= 1.0 for q in args.q_thresholds):
        raise ValueError("Every q threshold must be in [0, 1]")
    if not 0.0 < args.occupancy_probability_threshold <= 1.0:
        raise ValueError("occupancy probability threshold must be in (0, 1]")
    if not 0.0 <= args.overlap_contribution_threshold <= 1.0:
        raise ValueError("overlap contribution threshold must be in [0, 1]")
    run_dir = Path(args.run_dir)
    output_dir = Path(args.out_dir) if args.out_dir else run_dir / "ablation_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = Path(args.selected_views) if args.selected_views else run_dir / "selected_views.json"

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    scene, checkpoint = load_scene(args.checkpoint, device)
    config = checkpoint["arguments"]
    evaluation_preset = None
    if args.evaluation_preset_json:
        if args.eval_encounter_count is not None:
            raise ValueError(
                "Use either --evaluation_preset_json or --eval_encounter_count, not both"
            )
        evaluation_preset = load_encounter_batch_preset(
            args.evaluation_preset_json, allow_incomplete=True
        )
    encounter_count = (
        args.eval_encounter_count
        if args.eval_encounter_count is not None
        else config.get("encounter_count")
    )
    if evaluation_preset is not None:
        encounter_count = None
        encounter_names = evaluation_preset["encounter_names"]
    else:
        encounter_names = (
            None if args.eval_encounter_count is not None else config.get("encounter_names")
        )
    views, _ = load_dataset(
        args.data_root,
        image_size=config["image_size"],
        native_dr=config["native_dr"],
        native_dd=config["native_dd"],
        los_angle_deg=0.0,
        encounter_count=encounter_count,
        encounter_names=encounter_names,
        frames_per_encounter=config.get("frames_per_encounter"),
        frame_start=config.get("frame_start", 0),
        background_threshold=config.get("background_threshold", 0.0),
        normalization_percentile=config.get("normalization_percentile", 99.5),
        amplitude_processing_mode=config.get("amplitude_processing_mode", "control"),
        psf_beta_range=config.get("psf_beta_range", 3.0),
        psf_beta_doppler=config.get("psf_beta_doppler", 5.0),
        psf_mask_minimum_weight=config.get("psf_mask_minimum_weight", 0.25),
        clean_gain=config.get("clean_gain", 0.1),
        clean_maximum_removed_energy_fraction=config.get(
            "clean_maximum_removed_energy_fraction", 0.1
        ),
        legacy_background_threshold_for_decode=config.get(
            "legacy_background_threshold_for_decode", 0.1
        ),
    )
    training_keys = _training_view_keys(selected_path)
    if args.split == "heldout":
        views = [
            view
            for view in views
            if (view.encounter, view.frame_index) not in training_keys
        ]
        if not views:
            raise RuntimeError("No held-out frames remain; use --split all or retain fewer training frames")
    evaluation_views_path = (
        Path(args.evaluation_views_file) if args.evaluation_views_file else None
    )
    evaluation_views = _fixed_evaluation_views(
        views, evaluation_views_path, args.max_eval_views, args.eval_seed
    )
    image_metrics = _evaluate_images(
        scene,
        checkpoint,
        evaluation_views,
        device,
        output_dir,
        args.preview_count,
        args.background,
    )

    source_ply = run_dir / "gaussians_final.ply"
    if not source_ply.exists():
        export_ply(
            source_ply,
            scene,
            support_temperature=config.get("final_support_temperature", 1.0),
        )
    ply_metrics = _evaluate_ply(
        source_ply,
        output_dir,
        args.q_thresholds,
        args.graph_neighbors,
        args.connectivity_radius,
        args.outlier_multiplier,
        args.gt_geometry,
        args.gt_sample_count,
        args.distance_thresholds_m,
        args.eval_seed,
        args.prediction_unit_scale,
        args.reference_unit_scale,
        args.voxel_resolution,
        args.voxel_padding_fraction,
        args.gaussian_sigma_extent,
        not args.no_fill_prediction_voxels,
        args.occupancy_probability_threshold,
        args.overlap_contribution_threshold,
        not args.no_export_fused_ply,
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "run_dir": str(run_dir),
        "data_root": str(args.data_root),
        "evaluation_split": args.split,
        "training_view_count": len(training_keys),
        "candidate_evaluation_view_count": len(views),
        "evaluated_view_count": len(evaluation_views),
        "eval_encounter_count": encounter_count,
        "eval_seed": args.eval_seed,
        "evaluation_views_file": str(evaluation_views_path) if evaluation_views_path else None,
        "evaluation_preset": evaluation_preset,
        "geometry_coordinate_alignment": "none",
        "prediction_unit_scale": args.prediction_unit_scale,
        "reference_unit_scale": args.reference_unit_scale,
        "primary_geometry_metrics": ["voxel_iou_3d", "voxel_precision", "voxel_recall"],
        "voxel_resolution": args.voxel_resolution,
        "voxel_padding_fraction": args.voxel_padding_fraction,
        "gaussian_sigma_extent": args.gaussian_sigma_extent,
        "prediction_voxelization_mode": "noisy_or_gaussian_support",
        "occupancy_probability_threshold": args.occupancy_probability_threshold,
        "overlap_contribution_threshold": args.overlap_contribution_threshold,
        "fill_prediction_voxels": not args.no_fill_prediction_voxels,
        "image_metrics": image_metrics,
        "ply_metrics": ply_metrics,
    }
    write_json(output_dir / "evaluation_metrics.json", report)
    print(
        f"views={image_metrics['views']} "
        f"PSNR={image_metrics['mean_psnr_db']:.4f}dB "
        f"SSIM={image_metrics['mean_ssim']:.5f} "
        f"MAE={image_metrics['mean_mae']:.6f}"
    )
    for threshold, metrics in ply_metrics["thresholds"].items():
        gt = metrics["ground_truth"]
        gt_summary = ""
        if gt.get("available"):
            voxel = metrics["voxel_metrics"]
            gt_summary = (
                f" IoU3D={voxel['iou_3d']:.6f}"
                f" voxel-P={voxel['precision']:.6f}"
                f" voxel-R={voxel['recall']:.6f}"
                f" CD-L1={gt['chamfer_l1_m']:.6g}m"
                f" accuracy={gt['accuracy_mean_m']:.6g}m"
                f" completeness={gt['completeness_mean_m']:.6g}m"
                f" HD95={gt['hausdorff_p95_m']:.6g}m"
            )
        print(
            f"q>={threshold}: points={metrics['point_count']} "
            f"retained={metrics['retained_fraction']:.4f} "
            f"outliers={metrics['knn_outlier_ratio']} "
            f"largest_component={metrics['largest_component_ratio']}"
            f"{gt_summary}"
        )


if __name__ == "__main__":
    main()
