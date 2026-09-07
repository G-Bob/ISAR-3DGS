from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path


EXPERIMENTS: dict[str, dict[str, object]] = {
    "full": {
        "description": "Complete proposed method",
        "overrides": {},
    },
    "no_visibility": {
        "description": "Remove radar LOS visibility",
        "flags": ["--disable_visibility"],
        "overrides": {},
    },
    "no_support_sparsity": {
        "description": "Remove Gaussian target-support sparsity",
        "overrides": {"lambda_support_sparsity": 0.0},
    },
    "no_support_graph": {
        "description": "Remove local target-support graph consistency",
        "overrides": {"lambda_support_graph": 0.0},
    },
    "no_surface": {
        "description": "Remove all surface regularization",
        "overrides": {"lambda_surface": 0.0},
    },
    "no_adaptive": {
        "description": "Disable Gaussian split and prune",
        "overrides": {"adaptive_interval": 0},
    },
    "encounters_4": {
        "description": "Data-quantity ablation using four encounters",
        "overrides": {"encounter_count": 4},
    },
}

DEFAULT_EXPERIMENTS = ",".join(EXPERIMENTS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sequentially train and evaluate the radar-GS ablation study"
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_root", default="outputs/ablation_10enc")
    parser.add_argument("--config", default="configs/full_backprojection.json")
    parser.add_argument("--experiments", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--encounter_count", type=int, default=10)
    parser.add_argument("--frames_per_encounter", type=int, default=120)
    parser.add_argument("--los_angle_deg", type=float, default=5.0)
    parser.add_argument("--views_per_batch", type=int, default=16)
    parser.add_argument("--iters", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_eval_views", type=int, default=200)
    parser.add_argument("--eval_seed", type=int, default=12345)
    parser.add_argument("--preview_count", type=int, default=8)
    parser.add_argument("--gt_geometry")
    parser.add_argument("--voxel_resolution", type=int, default=256)
    parser.add_argument("--voxel_padding_fraction", type=float, default=0.05)
    parser.add_argument("--gaussian_sigma_extent", type=float, default=2.0)
    parser.add_argument("--baseline_run", help="Existing full-method run directory to evaluate")
    parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument("--evaluate_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def _command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(item) for item in command)


def _flatten(prefix: str, value: object, output: dict[str, object]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), item, output)
    elif isinstance(value, list):
        output[prefix] = json.dumps(value, separators=(",", ":"))
    else:
        output[prefix] = value


def _summary_rows(output_root: Path, experiment_dirs: dict[str, Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, run_dir in experiment_dirs.items():
        metrics_path = run_dir / "ablation_evaluation" / "evaluation_metrics.json"
        if not metrics_path.exists():
            continue
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        image = report["image_metrics"]
        timing_path = run_dir / "timing_report.json"
        timing = (
            json.loads(timing_path.read_text(encoding="utf-8"))
            if timing_path.exists()
            else {}
        )
        for threshold, ply in report["ply_metrics"]["thresholds"].items():
            row: dict[str, object] = {
                "experiment": name,
                "description": EXPERIMENTS[name]["description"],
                "q_threshold": float(threshold),
                "evaluated_views": report["evaluated_view_count"],
                "training_views": timing.get("selected_views", report["training_view_count"]),
                "final_gaussians": timing.get(
                    "final_gaussians", report["ply_metrics"]["original_point_count"]
                ),
                "training_total_seconds": timing.get("total_seconds"),
                "seconds_per_iteration": timing.get("seconds_per_iteration"),
                "peak_gpu_allocated_gb": timing.get("peak_gpu_allocated_gb"),
                "mean_mae": image["mean_mae"],
                "mean_mse": image["mean_mse"],
                "mean_psnr_db": image["mean_psnr_db"],
                "mean_ssim": image["mean_ssim"],
                "mean_log_mae": image["mean_log_mae"],
            }
            intrinsic = {key: value for key, value in ply.items() if key != "ground_truth"}
            _flatten("ply", intrinsic, row)
            _flatten("gt", ply.get("ground_truth", {}), row)
            rows.append(row)
    if rows:
        fields = sorted({key for row in rows for key in row})
        preferred = [
            "experiment",
            "description",
            "q_threshold",
            "evaluated_views",
            "training_views",
            "final_gaussians",
            "training_total_seconds",
            "seconds_per_iteration",
            "peak_gpu_allocated_gb",
            "mean_psnr_db",
            "mean_ssim",
            "mean_mae",
            "mean_mse",
            "mean_log_mae",
            "ply.voxel_metrics.iou_3d",
            "ply.voxel_metrics.precision",
            "ply.voxel_metrics.recall",
        ]
        fields = preferred + [field for field in fields if field not in preferred]
        with (output_root / "ablation_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        (output_root / "ablation_summary.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )
    return rows


def _run(command: list[str], log_path: Path, cwd: Path, dry_run: bool) -> None:
    print(_command_text(command), flush=True)
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
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}; inspect {log_path}"
        )


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (project_root / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = Path(args.config)
    if not config.is_absolute():
        config = (project_root / config).resolve()
    names = [name.strip() for name in args.experiments.split(",") if name.strip()]
    unknown = [name for name in names if name not in EXPERIMENTS]
    if unknown:
        raise ValueError(f"Unknown experiments: {unknown}; choices are {sorted(EXPERIMENTS)}")

    experiment_dirs: dict[str, Path] = {}
    manifest: dict[str, object] = {
        "data_root": str(data_root),
        "output_root": str(output_root),
        "config": str(config),
        "experiments": {},
        "fixed_evaluation": {
            "split": "heldout",
            "eval_encounter_count": args.encounter_count,
            "max_eval_views": args.max_eval_views,
            "eval_seed": args.eval_seed,
            "q_thresholds": [0.1, 0.3, 0.5],
            "evaluation_views_file": str(output_root / "evaluation_views.json"),
        },
    }

    for name in names:
        definition = EXPERIMENTS[name]
        if name == "full" and args.baseline_run:
            run_dir = Path(args.baseline_run).expanduser().resolve()
        else:
            run_dir = output_root / name
        experiment_dirs[name] = run_dir
        checkpoint = run_dir / "checkpoints" / "final.pt"
        overrides = {
            "encounter_count": args.encounter_count,
            "frames_per_encounter": args.frames_per_encounter,
            "los_angle_deg": args.los_angle_deg,
            "views_per_batch": args.views_per_batch,
            "iters": args.iters,
            "seed": args.seed,
            "lambda_support_sparsity": 0.003,
            "lambda_support_graph": 0.005,
            "prune_support_threshold": 0.10,
            "prune_evidence_threshold": 0.0001,
            "prune_min_visible_fraction": 0.20,
            "adaptive_interval": 500,
            "split_support_threshold": 0.60,
            "maximum_gaussians": 20000,
            "preview_interval": 1000,
            "checkpoint_interval": 5000,
        }
        overrides.update(definition.get("overrides", {}))
        train_command = [
            args.python,
            "-u",
            str(project_root / "run_reconstruction.py"),
            "--config",
            str(config),
            "--method",
            "radargs",
            "--data_root",
            str(data_root),
            "--out_dir",
            str(run_dir),
            "--device",
            args.device,
        ]
        for key, value in overrides.items():
            train_command.extend((f"--{key}", str(value)))
        train_command.extend(definition.get("flags", []))
        eval_command = [
            args.python,
            "-u",
            str(project_root / "evaluate_ablation.py"),
            "--checkpoint",
            str(checkpoint),
            "--data_root",
            str(data_root),
            "--run_dir",
            str(run_dir),
            "--device",
            args.device,
            "--split",
            "heldout",
            "--eval_encounter_count",
            str(args.encounter_count),
            "--max_eval_views",
            str(args.max_eval_views),
            "--eval_seed",
            str(args.eval_seed),
            "--preview_count",
            str(args.preview_count),
            "--evaluation_views_file",
            str(output_root / "evaluation_views.json"),
            "--q_thresholds",
            "0.1",
            "0.3",
            "0.5",
            "--voxel_resolution",
            str(args.voxel_resolution),
            "--voxel_padding_fraction",
            str(args.voxel_padding_fraction),
            "--gaussian_sigma_extent",
            str(args.gaussian_sigma_extent),
        ]
        if args.gt_geometry:
            eval_command.extend(("--gt_geometry", str(Path(args.gt_geometry).expanduser().resolve())))
        manifest["experiments"][name] = {
            "description": definition["description"],
            "run_dir": str(run_dir),
            "overrides": overrides,
            "training_command": train_command,
            "evaluation_command": eval_command,
        }
        (output_root / "ablation_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        using_external_baseline = name == "full" and bool(args.baseline_run)
        if not args.evaluate_only and not using_external_baseline:
            if checkpoint.exists() and args.skip_completed:
                print(f"[{name}] training already complete; skipping", flush=True)
            elif checkpoint.exists():
                raise FileExistsError(
                    f"[{name}] checkpoint already exists at {checkpoint}; use "
                    "--skip_completed or choose a new --output_root"
                )
            else:
                print(f"[{name}] training", flush=True)
                _run(train_command, output_root / "logs" / f"{name}_train.log", project_root, args.dry_run)
        if not args.dry_run and not checkpoint.exists():
            raise FileNotFoundError(f"[{name}] final checkpoint not found: {checkpoint}")
        print(f"[{name}] evaluation", flush=True)
        _run(eval_command, output_root / "logs" / f"{name}_evaluation.log", project_root, args.dry_run)
        if not args.dry_run:
            _summary_rows(output_root, experiment_dirs)

    if args.dry_run:
        print(f"Dry run complete. Manifest: {output_root / 'ablation_manifest.json'}")
    else:
        rows = _summary_rows(output_root, experiment_dirs)
        print(f"Completed {len(names)} experiments and wrote {len(rows)} summary rows")
        print(output_root / "ablation_summary.csv")


if __name__ == "__main__":
    main()
