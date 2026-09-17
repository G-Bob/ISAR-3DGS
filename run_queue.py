"""Train and evaluate all skynet5d-4 K=1..8 jobs in parallel."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path("/root/autodl-fs/radar_gs_isar_enc/outputs/radar_gs/skynet5d4_k1_k8_v22_20260909")
DATA = Path("/root/Yao/datasets/skynet5d-4")
WORK = Path(__file__).resolve().parent
PYTHON = "/root/miniconda3/bin/python"


def write_status(job: dict, state: str, **extra: object) -> None:
    path = ROOT / "status" / f"{job['name']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(dict(job, state=state, **extra), indent=2))
    temp.replace(path)


def execute(job: dict) -> bool:
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    run = Path(job["run_dir"])
    run.mkdir(parents=True, exist_ok=True)
    final = run / "checkpoints/final.pt"
    if not final.exists():
        command = [PYTHON, "-u", "run_reconstruction.py", "--config", job["config"], "--data_root", str(DATA), "--out_dir", str(run)]
        with (run / "train.log").open("a") as stream:
            process = subprocess.Popen(command, cwd=WORK, env=env, stdout=stream, stderr=subprocess.STDOUT)
            write_status(job, "training", pid=process.pid, command=command, start=time.time())
            if process.wait() != 0:
                write_status(job, "training_failed")
                return False

    output = ROOT / "evaluation" / job["name"]
    output.mkdir(parents=True, exist_ok=True)
    metrics = output / "evaluation_metrics.json"
    if not metrics.exists():
        command = [PYTHON, "-u", "evaluate_ablation.py", "--checkpoint", str(final), "--data_root", str(DATA), "--run_dir", str(run), "--out_dir", str(output), "--device", "cuda", "--split", "all", "--max_eval_views", "240", "--eval_seed", "42", "--preview_count", "12", "--q_thresholds", "0.3", "0.5", "--no_export_fused_ply", "--gt_geometry", job["gt"]]
        with (output / "evaluation.log").open("a") as stream:
            process = subprocess.Popen(command, cwd=WORK, env=env, stdout=stream, stderr=subprocess.STDOUT)
            write_status(job, "evaluating_in_sample", pid=process.pid, command=command)
            if process.wait() != 0:
                write_status(job, "evaluation_failed")
                return False

    if not (output / "pointcloud_native_cad.png").exists():
        with (output / "render.log").open("a") as stream:
            code = subprocess.call([PYTHON, "render_sequential_result.py", "--run", str(run), "--out", str(output)], cwd=WORK, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if code:
            write_status(job, "render_failed")
            return False
    write_status(job, "complete", finish=time.time())
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()
    jobs = json.loads((ROOT / "jobs.json").read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        results = [future.result() for future in concurrent.futures.as_completed([pool.submit(execute, job) for job in jobs])]
    (ROOT / ("ALL_COMPLETE" if all(results) else "QUEUE_FINISHED_WITH_ERRORS")).touch()


if __name__ == "__main__":
    main()
