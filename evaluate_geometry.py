from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from radar_gs.ply_evaluation import (
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare a q-thresholded Radar-GS PLY with reference 3D geometry"
    )
    parser.add_argument("--prediction_ply", required=True)
    parser.add_argument("--gt_geometry", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--q_thresholds", nargs="+", type=float, default=[0.1, 0.3, 0.5])
    parser.add_argument(
        "--distance_thresholds_m", nargs="+", type=float, default=[0.05, 0.10, 0.20]
    )
    parser.add_argument("--gt_sample_count", type=int, default=100000)
    parser.add_argument("--prediction_unit_scale", type=float, default=1.0)
    parser.add_argument("--reference_unit_scale", type=float, default=1.0)
    parser.add_argument("--voxel_resolution", type=int, default=256)
    parser.add_argument("--voxel_padding_fraction", type=float, default=0.05)
    parser.add_argument("--gaussian_sigma_extent", type=float, default=2.0)
    parser.add_argument(
        "--occupancy_probability_threshold",
        type=float,
        default=0.5,
        help="Noisy-OR support probability required for a predicted occupied voxel",
    )
    parser.add_argument(
        "--overlap_contribution_threshold",
        type=float,
        default=0.05,
        help="Minimum q-weighted Gaussian contribution counted in overlap previews",
    )
    parser.add_argument(
        "--no_fill_prediction_voxels",
        action="store_true",
        help="Keep only Gaussian ellipsoid voxels instead of filling enclosed interiors",
    )
    parser.add_argument(
        "--no_export_fused_ply",
        action="store_true",
        help="Do not export the boundary mesh of the accumulated occupancy volume",
    )
    parser.add_argument(
        "--save_voxel_grids",
        action="store_true",
        help="Also save compressed probability, overlap-count and occupancy arrays",
    )
    parser.add_argument("--graph_neighbors", type=int, default=8)
    parser.add_argument("--connectivity_radius", type=float, default=0.0)
    parser.add_argument("--outlier_multiplier", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=12345)
    return parser


def _threshold_label(value: float) -> str:
    return f"q{int(round(value * 100)):03d}"


def _csv_row(q: float, metrics: dict) -> dict:
    row = {
        "q_threshold": q,
        "point_count": metrics["point_count"],
        "retained_fraction": metrics["retained_fraction"],
        "knn_outlier_ratio": metrics["knn_outlier_ratio"],
        "largest_component_ratio": metrics["largest_component_ratio"],
        "voxel_iou_3d": metrics["voxel_metrics"]["iou_3d"],
        "voxel_precision": metrics["voxel_metrics"]["precision"],
        "voxel_recall": metrics["voxel_metrics"]["recall"],
        "voxel_f_score": metrics["voxel_metrics"]["f_score"],
        "occupancy_probability_threshold": metrics["voxel_metrics"].get(
            "occupancy_probability_threshold"
        ),
        "multi_gaussian_overlap_fraction": metrics["support_volume_summary"][
            "multi_gaussian_overlap_fraction"
        ],
        "maximum_overlap_count": metrics["support_volume_summary"][
            "maximum_overlap_count"
        ],
    }
    gt = metrics["ground_truth"]
    for name in (
        "accuracy_mean_m",
        "accuracy_median_m",
        "accuracy_p95_m",
        "completeness_mean_m",
        "completeness_median_m",
        "completeness_p95_m",
        "chamfer_l1_m",
        "chamfer_l2_rmse_m",
        "hausdorff_p95_m",
        "hausdorff_m",
        "normalized_chamfer_l1",
        "normalized_hausdorff_p95",
        "normal_consistency_abs_cosine",
    ):
        row[name] = gt.get(name)
    for threshold, score in gt.get("distance_threshold_scores", {}).items():
        suffix = threshold.replace(".", "p")
        row[f"precision_at_{suffix}m"] = score["precision"]
        row[f"recall_at_{suffix}m"] = score["recall"]
        row[f"fscore_at_{suffix}m"] = score["f_score"]
    return row


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

    prediction_path = Path(args.prediction_ply)
    reference_path = Path(args.gt_geometry)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = read_ascii_ply(prediction_path)
    original_count = len(table.values)
    reference_points, reference_normals = load_reference_geometry(
        reference_path, sample_count=args.gt_sample_count, seed=args.seed
    )
    reference_points = reference_points * args.reference_unit_scale
    voxel_spec = make_reference_voxel_grid(
        reference_points,
        resolution=args.voxel_resolution,
        padding_fraction=args.voxel_padding_fraction,
    )
    reference_voxels, reference_voxel_mode = voxelize_reference_geometry(
        reference_path,
        voxel_spec,
        unit_scale=args.reference_unit_scale,
        fallback_points=reference_points,
    )

    threshold_reports: dict[str, dict] = {}
    rows: list[dict] = []
    for q in args.q_thresholds:
        filtered = filter_by_support(table, q)
        filtered_path = output_dir / f"gaussians_{_threshold_label(q)}.ply"
        write_ascii_ply(filtered_path, filtered)
        points = np.column_stack([filtered.column(axis) for axis in ("x", "y", "z")])
        points = points * args.prediction_unit_scale
        support = filtered.column("target_support_probability")
        normals = None
        if all(name in filtered.properties for name in ("nx", "ny", "nz")):
            normals = np.column_stack([filtered.column(axis) for axis in ("nx", "ny", "nz")])
        if normals is None or not all(
            name in filtered.properties for name in ("scale_t1", "scale_t2", "scale_normal")
        ):
            raise ValueError(
                "G2-SDF voxel comparison requires nx/ny/nz and "
                "scale_t1/scale_t2/scale_normal in the prediction PLY"
            )
        scales = np.column_stack(
            [filtered.column(name) for name in ("scale_t1", "scale_t2", "scale_normal")]
        ) * args.prediction_unit_scale
        support_volume = voxelize_gaussian_support(
            points,
            normals,
            scales,
            support,
            voxel_spec,
            sigma_extent=args.gaussian_sigma_extent,
            occupancy_probability_threshold=args.occupancy_probability_threshold,
            overlap_contribution_threshold=args.overlap_contribution_threshold,
            fill_interior=not args.no_fill_prediction_voxels,
        )
        voxel_metrics = voxel_overlap_metrics(
            support_volume.occupied_grid,
            reference_voxels,
            voxel_spec,
            prediction_outside_count=support_volume.outside_occupied_count,
            reference_mode=reference_voxel_mode,
            sigma_extent=args.gaussian_sigma_extent,
            fill_prediction_interior=not args.no_fill_prediction_voxels,
            prediction_voxelization_mode="noisy_or_gaussian_support",
            occupancy_probability_threshold=args.occupancy_probability_threshold,
        )
        preview_path = output_dir / f"gaussian_overlap_{_threshold_label(q)}.png"
        save_gaussian_support_preview(
            preview_path,
            support_volume,
            q_threshold=q,
            occupancy_probability_threshold=args.occupancy_probability_threshold,
        )
        fused_ply_path = None
        if not args.no_export_fused_ply:
            fused_ply_path = output_dir / f"gaussian_fused_{_threshold_label(q)}.ply"
            write_voxel_surface_ply(
                fused_ply_path,
                support_volume.occupied_grid,
                voxel_spec,
            )
        voxel_grid_path = None
        if args.save_voxel_grids:
            voxel_grid_path = output_dir / f"gaussian_volume_{_threshold_label(q)}.npz"
            np.savez_compressed(
                voxel_grid_path,
                probability=support_volume.probability_grid,
                overlap_count=support_volume.overlap_count_grid,
                occupied=support_volume.occupied_grid,
                origin_m=voxel_spec.origin_m,
                voxel_size_m=voxel_spec.voxel_size_m,
            )

        metrics = intrinsic_point_metrics(
            points,
            support,
            graph_neighbors=args.graph_neighbors,
            connectivity_radius=args.connectivity_radius,
            outlier_multiplier=args.outlier_multiplier,
        )
        metrics.update(
            {
                "q_threshold": float(q),
                "retained_fraction": float(len(points) / max(original_count, 1)),
                "filtered_ply": str(filtered_path),
                "overlap_preview": str(preview_path),
                "fused_occupancy_ply": str(fused_ply_path) if fused_ply_path else None,
                "voxel_grid_npz": str(voxel_grid_path) if voxel_grid_path else None,
                "support_volume_summary": support_volume.summary(),
                "voxel_metrics": voxel_metrics,
                "ground_truth": ground_truth_metrics(
                    points,
                    reference_points,
                    args.distance_thresholds_m,
                    prediction_normals=normals,
                    reference_normals=reference_normals,
                ),
            }
        )
        threshold_reports[f"{q:g}"] = metrics
        rows.append(_csv_row(q, metrics))

    report = {
        "prediction_ply": str(prediction_path),
        "gt_geometry": str(reference_path),
        "coordinate_alignment": "none",
        "warning": (
            "Prediction and reference must already share the same coordinate frame, "
            "orientation, origin and metric scale. No ICP or similarity alignment was applied."
        ),
        "prediction_unit_scale": args.prediction_unit_scale,
        "reference_unit_scale": args.reference_unit_scale,
        "reference_sample_count": int(len(reference_points)),
        "original_point_count": original_count,
        "distance_thresholds_m": args.distance_thresholds_m,
        "primary_comparison_protocol": "G2-SDF voxel IoU/Precision/Recall",
        "voxel_grid_shared_across_q_thresholds": True,
        "prediction_voxelization_mode": "noisy_or_gaussian_support",
        "occupancy_probability_threshold": args.occupancy_probability_threshold,
        "overlap_contribution_threshold": args.overlap_contribution_threshold,
        "thresholds": threshold_reports,
    }
    (output_dir / "geometry_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (output_dir / "geometry_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    primary_fields = [
        "q_threshold",
        "occupancy_probability_threshold",
        "voxel_iou_3d",
        "voxel_precision",
        "voxel_recall",
    ]
    with (output_dir / "g2sdf_primary_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=primary_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in primary_fields} for row in rows)

    for row in rows:
        if row["chamfer_l1_m"] is None:
            print(f"q>={row['q_threshold']:g}: points=0; geometric metrics unavailable")
            continue
        print(
            f"q>={row['q_threshold']:g}: points={row['point_count']} "
            f"IoU3D={row['voxel_iou_3d']:.6f} "
            f"voxel-P={row['voxel_precision']:.6f} "
            f"voxel-R={row['voxel_recall']:.6f} "
            f"CD-L1={row['chamfer_l1_m']:.6g}m "
            f"accuracy={row['accuracy_mean_m']:.6g}m "
            f"completeness={row['completeness_mean_m']:.6g}m "
            f"HD95={row['hausdorff_p95_m']:.6g}m"
        )
    print(f"Reports written to {output_dir}")


if __name__ == "__main__":
    main()
