"""Run the skynet5d-4 continual chain serially and evaluate every stage."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


BASE = Path("/root/autodl-fs/radar_gs_isar_enc")
ROOT = BASE / "outputs/radar_gs/skynet5d4_incremental_v23_20260909"
DATA = Path("/root/Yao/datasets/skynet5d-4")
WORK = Path(__file__).resolve().parent
PYTHON = "/root/miniconda3/bin/python"


def status(job: dict, state: str, **extra: object) -> None:
    path = ROOT / "status" / f"{job['name']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(dict(job, state=state, **extra), indent=2))
    temporary.replace(path)


def call(command: list[str], log: Path, env: dict, job: dict, state: str) -> bool:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        process = subprocess.Popen(command, cwd=WORK, env=env, stdout=stream, stderr=subprocess.STDOUT)
        status(job, state, pid=process.pid, command=command, timestamp=time.time())
        return process.wait() == 0


def evaluate(job: dict, env: dict, mode: str) -> bool:
    run = Path(job["run_dir"])
    out = ROOT / ("evaluation_current" if mode == "current" else "evaluation_cumulative") / job["name"]
    metrics = out / "evaluation_metrics.json"
    if metrics.exists():
        return True
    command = [PYTHON, "-u", "evaluate_ablation.py", "--checkpoint", str(run / "checkpoints/final.pt"), "--data_root", str(DATA), "--run_dir", str(run), "--out_dir", str(out), "--device", "cuda", "--split", "all", "--max_eval_views", "240", "--eval_seed", "42", "--preview_count", "12", "--q_thresholds", "0.3", "0.5", "--no_export_fused_ply", "--gt_geometry", job["gt"]]
    if mode == "cumulative":
        command += ["--evaluation_preset_json", job["cumulative_preset"]]
    return call(command, out / "evaluation.log", env, job, f"evaluating_{mode}")


def main() -> None:
    jobs = json.loads((ROOT / "jobs.json").read_text())
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    success = True
    for job in jobs:
        run = Path(job["run_dir"])
        final = run / "checkpoints/final.pt"
        if not final.exists():
            command = [PYTHON, "-u", "run_reconstruction.py", "--config", job["config"], "--data_root", str(DATA), "--out_dir", str(run)]
            if not call(command, run / "train.log", env, job, "training"):
                status(job, "training_failed")
                success = False
                break
        if not evaluate(job, env, "current") or not evaluate(job, env, "cumulative"):
            status(job, "evaluation_failed")
            success = False
            break
        cumulative_out = ROOT / "evaluation_cumulative" / job["name"]
        if not (cumulative_out / "pointcloud_native_cad.png").exists():
            if not call([PYTHON, "render_sequential_result.py", "--run", str(run), "--out", str(cumulative_out)], cumulative_out / "render.log", env, job, "rendering"):
                status(job, "render_failed")
                success = False
                break
        status(job, "complete", finish=time.time())
    (ROOT / ("ALL_COMPLETE" if success else "CHAIN_STOPPED_WITH_ERROR")).touch()


if __name__ == "__main__":
    main()
