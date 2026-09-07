from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train independently optimized fixed-K encounter subsets"
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--selection_dir", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--gt_geometry")
    parser.add_argument("--config", default="configs/full_backprojection.json")
    parser.add_argument("--counts", nargs="+", type=int)
    parser.add_argument("--images_per_encounter", type=int, required=True)
    parser.add_argument("--frames_per_encounter", type=int, default=120)
    parser.add_argument("--views_per_batch", type=int, default=16)
    parser.add_argument("--iters", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--q_thresholds", nargs="+", type=float, default=[0.1, 0.3, 0.5])
    parser.add_argument("--decision_q", type=float, default=0.3)
    parser.add_argument("--decision_distance_m", type=float, default=0.1)
    parser.add_argument("--coverage_ratio", type=float, default=0.95)
    parser.add_argument("--observability_ratio", type=float, default=0.95)
    parser.add_argument("--fscore_ratio", type=float, default=0.95)
    parser.add_argument("--chamfer_ratio", type=float, default=1.05)
    parser.add_argument("--voxel_iou_ratio", type=float, default=0.95)
    parser.add_argument("--voxel_precision_ratio", type=float, default=0.95)
    parser.add_argument("--voxel_recall_ratio", type=float, default=0.95)
    parser.add_argument("--voxel_resolution", type=int, default=256)
    parser.add_argument("--voxel_padding_fraction", type=float, default=0.05)
    parser.add_argument("--gaussian_sigma_extent", type=float, default=2.0)
    parser.add_argument("--ssim_ratio", type=float, default=0.95)
    parser.add_argument("--minimum_largest_component_ratio", type=float, default=0.90)
    parser.add_argument("--max_eval_views", type=int, default=200)
    parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument("--evaluate_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def _run(command: list[str], log_path: Path, cwd: Path, dry_run: bool) -> None:
    print(" ".join(shlex.quote(value) for value in command), flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); inspect {log_path}")


def _read_subsets(selection_dir: Path) -> dict[int, list[str]]:
    subset_dir = selection_dir / "subsets"
    subsets: dict[int, list[str]] = {}
    for path in sorted(subset_dir.glob("k*.txt")):
        try:
            count = int(path.stem[1:])
        except ValueError:
            continue
        names = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(names) != count:
            raise ValueError(f"{path} contains {len(names)} encounters, expected {count}")
        if len(set(names)) != count:
            raise ValueError(f"{path} contains duplicate encounter names")
        subsets[count] = names
    if not subsets:
        raise FileNotFoundError(
            f"No fixed-K subset files found under {subset_dir}; run select_encounters.py first"
        )
    return subsets


def _read_coverage(selection_dir: Path) -> dict[int, dict[str, float]]:
    with (selection_dir / "fixed_k_selection.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {
        int(row["encounter_count"]): {
            key: float(value)
            for key, value in row.items()
            if key not in {"encounter_count", "encounters", "search_method"}
        }
        for row in rows
    }


def _threshold_key(value: float) -> str:
    return f"{float(value):g}"


def _collect_rows(
    counts: list[int],
    run_dirs: dict[int, Path],
    coverage: dict[int, dict[str, float]],
) -> list[dict]:
    rows = []
    for count in counts:
        metrics_path = run_dirs[count] / "ablation_evaluation" / "evaluation_metrics.json"
        if not metrics_path.exists():
            continue
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        image = report["image_metrics"]
        timing_path = run_dirs[count] / "timing_report.json"
        timing = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.exists() else {}
        for q_key, ply in report["ply_metrics"]["thresholds"].items():
            row = {
                "encounter_count": count,
                "q_threshold": float(q_key),
                "direction_union_pct": coverage[count]["direction_union_pct"],
                "normalized_minimum_eigenvalue": coverage[count][
                    "normalized_minimum_eigenvalue"
                ],
                "observability_isotropy": coverage[count]["observability_isotropy"],
                "information_condition_number": coverage[count][
                    "information_condition_number"
                ],
                "mean_center_separation_deg": coverage[count][
                    "mean_center_separation_deg"
                ],
                "evaluated_views": report["evaluated_view_count"],
                "mean_psnr_db": image["mean_psnr_db"],
                "mean_ssim": image["mean_ssim"],
                "mean_mae": image["mean_mae"],
                "point_count": ply["point_count"],
                "retained_fraction": ply["retained_fraction"],
                "knn_outlier_ratio": ply["knn_outlier_ratio"],
                "largest_component_ratio": ply["largest_component_ratio"],
                "training_total_seconds": timing.get("total_seconds"),
                "final_gaussians": timing.get("final_gaussians"),
            }
            gt = ply.get("ground_truth", {})
            voxel = ply.get("voxel_metrics", {})
            row["voxel_iou_3d"] = voxel.get("iou_3d")
            row["voxel_precision"] = voxel.get("precision")
            row["voxel_recall"] = voxel.get("recall")
            row["voxel_f_score"] = voxel.get("f_score")
            row["chamfer_l1_m"] = gt.get("chamfer_l1_m")
            row["accuracy_mean_m"] = gt.get("accuracy_mean_m")
            row["completeness_mean_m"] = gt.get("completeness_mean_m")
            for distance, score in gt.get("distance_threshold_scores", {}).items():
                row[f"precision_at_{distance}m"] = score["precision"]
                row[f"recall_at_{distance}m"] = score["recall"]
                row[f"f_score_at_{distance}m"] = score["f_score"]
            rows.append(row)
    return rows


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    extra = sorted({key for row in rows for key in row} - set(fields))
    fields.extend(extra)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _recommend(
    args,
    rows: list[dict],
    coverage: dict[int, dict[str, float]],
    use_ssim: bool,
) -> dict:
    q_key = float(args.decision_q)
    candidates = [row for row in rows if abs(row["q_threshold"] - q_key) < 1.0e-9]
    if not candidates:
        return {"available": False, "reason": f"No rows for q={q_key}"}
    maximum_count = max(coverage)
    full_direction_coverage = coverage[maximum_count]["direction_union_pct"]
    full_observability = coverage[maximum_count]["normalized_minimum_eigenvalue"]
    best_ssim = max(row["mean_ssim"] for row in candidates)
    gt_candidates = [row for row in candidates if row.get("chamfer_l1_m") is not None]
    voxel_candidates = [row for row in candidates if row.get("voxel_iou_3d") is not None]
    best_voxel_iou = (
        max(row["voxel_iou_3d"] for row in voxel_candidates) if voxel_candidates else None
    )
    best_voxel_precision = (
        max(row["voxel_precision"] for row in voxel_candidates) if voxel_candidates else None
    )
    best_voxel_recall = (
        max(row["voxel_recall"] for row in voxel_candidates) if voxel_candidates else None
    )
    best_chamfer = min(row["chamfer_l1_m"] for row in gt_candidates) if gt_candidates else None
    fscore_column = f"f_score_at_{_threshold_key(args.decision_distance_m)}m"
    best_fscore = (
        max(row.get(fscore_column, 0.0) for row in gt_candidates)
        if gt_candidates
        else None
    )
    decisions = []
    for row in sorted(candidates, key=lambda item: item["encounter_count"]):
        checks = {
            "observation_direction_coverage": row["direction_union_pct"]
            >= args.coverage_ratio * full_direction_coverage,
            "projection_observability": row["normalized_minimum_eigenvalue"]
            >= args.observability_ratio * full_observability,
            "largest_component": (
                row["largest_component_ratio"] is not None
                and row["largest_component_ratio"] >= args.minimum_largest_component_ratio
            ),
        }
        if use_ssim:
            checks["heldout_ssim"] = row["mean_ssim"] >= args.ssim_ratio * best_ssim
        if best_voxel_iou is not None:
            checks["voxel_iou_3d"] = (
                row.get("voxel_iou_3d", 0.0) >= args.voxel_iou_ratio * best_voxel_iou
            )
            checks["voxel_precision"] = (
                row.get("voxel_precision", 0.0)
                >= args.voxel_precision_ratio * best_voxel_precision
            )
            checks["voxel_recall"] = (
                row.get("voxel_recall", 0.0) >= args.voxel_recall_ratio * best_voxel_recall
            )
        decisions.append(
            {
                "encounter_count": row["encounter_count"],
                "passes": all(checks.values()),
                "checks": checks,
            }
        )
    passed = [item["encounter_count"] for item in decisions if item["passes"]]
    recommendation = min(passed) if passed else None
    return {
        "available": recommendation is not None,
        "recommended_minimum_encounters": recommendation,
        "decision_q_threshold": args.decision_q,
        "decision_distance_m": args.decision_distance_m,
        "criteria": {
            "observation_direction_fraction_of_full": args.coverage_ratio,
            "projection_observability_fraction_of_full": args.observability_ratio,
            "ssim_fraction_of_best": args.ssim_ratio if use_ssim else None,
            "minimum_largest_component_ratio": args.minimum_largest_component_ratio,
            "voxel_iou_fraction_of_best": args.voxel_iou_ratio,
            "voxel_precision_fraction_of_best": args.voxel_precision_ratio,
            "voxel_recall_fraction_of_best": args.voxel_recall_ratio,
        },
        "supplementary_diagnostics": {
            "chamfer_multiplier_of_best": args.chamfer_ratio,
            "distance_fscore_fraction_of_best": args.fscore_ratio,
            "distance_fscore_threshold_m": args.decision_distance_m,
            "participates_in_primary_decision": False,
        },
        "best_observed": {
            "full_set_direction_union_pct": full_direction_coverage,
            "full_set_normalized_minimum_eigenvalue": full_observability,
            "mean_ssim": best_ssim,
            "chamfer_l1_m": best_chamfer,
            "f_score": best_fscore,
            "voxel_iou_3d": best_voxel_iou,
            "voxel_precision": best_voxel_precision,
            "voxel_recall": best_voxel_recall,
        },
        "decisions": decisions,
        "validation_note": (
            "Repeat K-1, K and K+1 with at least three random seeds before reporting "
            "the minimum encounter count."
        ),
        "selection_note": (
            "Each fixed-K subset was optimized independently using only view_angle.npy "
            "acquisition geometry. The target mesh and ISAR intensity were not used. GT "
            "metrics, when provided, are post-training evaluation only."
        ),
        "image_metric_note": (
            "Held-out SSIM participates in the decision."
            if use_ssim
            else "All frames were used for training, so image metrics are reconstruction-fit "
            "diagnostics and do not participate in the minimum-K decision."
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    data_root = Path(args.data_root).expanduser().resolve()
    selection_dir = Path(args.selection_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = Path(args.config)
    if not config.is_absolute():
        config = (project_root / config).resolve()
    subsets = _read_subsets(selection_dir)
    coverage = _read_coverage(selection_dir)
    counts = sorted(set(args.counts or subsets))
    missing_counts = [count for count in counts if count not in subsets or count not in coverage]
    if missing_counts:
        raise ValueError(
            f"No independently optimized subset/metrics for K={missing_counts}; "
            "rerun select_encounters.py with these --counts"
        )

    source_eval = selection_dir / "evaluation_views.json"
    source_report = json.loads(source_eval.read_text(encoding="utf-8"))
    selection_report = json.loads(
        (selection_dir / "selection_report.json").read_text(encoding="utf-8")
    )
    full_encounter_count = int(selection_report["encounter_count"])
    has_heldout = bool(source_report.get("views"))
    evaluation_split = "heldout" if has_heldout else "all"
    shared_eval = source_eval if has_heldout else output_root / "evaluation_views_all.json"
    run_dirs: dict[int, Path] = {}
    manifest = {
        "data_root": str(data_root),
        "selection_dir": str(selection_dir),
        "gt_geometry": str(Path(args.gt_geometry).expanduser().resolve()) if args.gt_geometry else None,
        "counts": counts,
        "images_per_encounter": args.images_per_encounter,
        "evaluation_split": evaluation_split,
        "runs": {},
    }

    for count in counts:
        run_dir = output_root / f"k{count:02d}"
        run_dirs[count] = run_dir
        subset_file = selection_dir / "subsets" / f"k{count:02d}.txt"
        checkpoint = run_dir / "checkpoints" / "final.pt"
        train = [
            args.python,
            "-u",
            str(project_root / "run_reconstruction.py"),
            "--config",
            str(config),
            "--data_root",
            str(data_root),
            "--out_dir",
            str(run_dir),
            "--encounter_names_file",
            str(subset_file),
            "--frames_per_encounter",
            str(args.frames_per_encounter),
            "--images_per_encounter",
            str(args.images_per_encounter),
            "--los_angle_deg",
            "0",
            "--views_per_batch",
            str(args.views_per_batch),
            "--iters",
            str(args.iters),
            "--seed",
            str(args.seed),
            "--lambda_support_sparsity",
            "0.003",
            "--lambda_support_graph",
            "0.005",
            "--prune_support_threshold",
            "0.10",
            "--prune_min_visible_fraction",
            "0.20",
            "--adaptive_interval",
            "500",
            "--device",
            args.device,
        ]
        evaluate = [
            args.python,
            "-u",
            str(project_root / "evaluate_ablation.py"),
            "--checkpoint",
            str(checkpoint),
            "--data_root",
            str(data_root),
            "--run_dir",
            str(run_dir),
            "--split",
            evaluation_split,
            "--eval_encounter_count",
            str(full_encounter_count),
            "--max_eval_views",
            str(args.max_eval_views),
            "--evaluation_views_file",
            str(shared_eval),
            "--q_thresholds",
            *[str(value) for value in args.q_thresholds],
            "--device",
            args.device,
            "--voxel_resolution",
            str(args.voxel_resolution),
            "--voxel_padding_fraction",
            str(args.voxel_padding_fraction),
            "--gaussian_sigma_extent",
            str(args.gaussian_sigma_extent),
        ]
        if args.gt_geometry:
            evaluate.extend(("--gt_geometry", str(Path(args.gt_geometry).expanduser().resolve())))
        manifest["runs"][str(count)] = {
            "encounters": subsets[count],
            "training_command": train,
            "evaluation_command": evaluate,
        }
        (output_root / "encounter_count_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        if not args.evaluate_only:
            if checkpoint.exists() and args.skip_completed:
                print(f"[K={count}] training complete; skipping")
            elif checkpoint.exists():
                raise FileExistsError(
                    f"Checkpoint exists: {checkpoint}; use --skip_completed or a new output root"
                )
            else:
                print(f"[K={count}] training")
                _run(train, output_root / "logs" / f"k{count:02d}_train.log", project_root, args.dry_run)
        if not args.dry_run and not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        print(f"[K={count}] evaluation")
        _run(evaluate, output_root / "logs" / f"k{count:02d}_evaluation.log", project_root, args.dry_run)
        if not args.dry_run:
            rows = _collect_rows(counts, run_dirs, coverage)
            _write_rows(output_root / "encounter_count_summary.csv", rows)
            recommendation = _recommend(args, rows, coverage, use_ssim=has_heldout)
            (output_root / "minimum_encounter_recommendation.json").write_text(
                json.dumps(recommendation, indent=2), encoding="utf-8"
            )

    if args.dry_run:
        print(output_root / "encounter_count_manifest.json")
    else:
        rows = _collect_rows(counts, run_dirs, coverage)
        _write_rows(output_root / "encounter_count_summary.csv", rows)
        recommendation = _recommend(args, rows, coverage, use_ssim=has_heldout)
        (output_root / "minimum_encounter_recommendation.json").write_text(
            json.dumps(recommendation, indent=2), encoding="utf-8"
        )
        print(json.dumps(recommendation, indent=2))


if __name__ == "__main__":
    main()
