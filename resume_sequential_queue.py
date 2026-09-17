"""Resume the sequential experiment queue without interrupting live trainers."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/root/autodl-fs/radar_gs_isar_enc/outputs/radar_gs/skynet5d3_sequential_v21_20260907')
WORK = Path(__file__).resolve().parent
PYTHON = '/root/miniconda3/bin/python'


def read_status(name: str) -> dict:
    path = ROOT / 'status' / f'{name}.json'
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {'name': name, 'state': 'queued'}


def write_status(job: dict, state: str, **extra) -> None:
    path = ROOT / 'status' / f'{job["name"]}.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(job, state=state, **extra), indent=2))
    temporary.replace(path)


def process_alive(pid: int) -> bool:
    stat = Path(f'/proc/{pid}/stat')
    if not stat.exists():
        return False
    try:
        return stat.read_text().split()[2] != 'Z'
    except OSError:
        return False


def wait_for_existing_training(job: dict, pid: int) -> bool:
    while process_alive(pid):
        time.sleep(10)
    run = Path(job['run_dir'])
    return (run / 'checkpoints/final.pt').exists()


def launch_training(job: dict, environment: dict) -> bool:
    run = Path(job['run_dir'])
    run.mkdir(parents=True, exist_ok=True)
    log = run / 'train.log'
    if log.exists():
        return (run / 'checkpoints/final.pt').exists()
    command = [
        PYTHON, '-u', 'run_reconstruction.py', '--config', job['config'],
        '--data_root', f'/root/Yao/datasets/skynet5d-3/{job["scenario"]}',
        '--out_dir', str(run),
    ]
    with log.open('x') as stream:
        process = subprocess.Popen(
            command, cwd=WORK, env=environment, stdout=stream,
            stderr=subprocess.STDOUT,
        )
        write_status(job, 'training', pid=process.pid, start=time.time(), command=command)
        return process.wait() == 0


def evaluate_and_render(job: dict, environment: dict) -> bool:
    run = Path(job['run_dir'])
    output = ROOT / 'evaluation' / job['name']
    output.mkdir(parents=True, exist_ok=True)
    metrics = output / 'evaluation_metrics.json'
    if not metrics.exists():
        command = [
            PYTHON, '-u', 'evaluate_ablation.py',
            '--checkpoint', str(run / 'checkpoints/final.pt'),
            '--data_root', f'/root/Yao/datasets/skynet5d-3/{job["scenario"]}',
            '--run_dir', str(run), '--out_dir', str(output), '--device', 'cuda',
            '--split', 'all', '--max_eval_views', '240', '--eval_seed', '42',
            '--preview_count', '12', '--q_thresholds', '0.3', '0.5',
            '--no_export_fused_ply', '--gt_geometry', job['gt'],
        ]
        with (output / 'evaluation.log').open('a') as stream:
            process = subprocess.Popen(
                command, cwd=WORK, env=environment, stdout=stream,
                stderr=subprocess.STDOUT,
            )
            write_status(job, 'evaluating_in_sample', pid=process.pid, command=command)
            if process.wait() != 0:
                return False
    if not (output / 'pointcloud_native_cad.png').exists():
        with (output / 'render.log').open('a') as stream:
            code = subprocess.call(
                [PYTHON, 'render_sequential_result.py', '--run', str(run), '--out', str(output)],
                cwd=WORK, env=environment, stdout=stream, stderr=subprocess.STDOUT,
            )
        if code:
            return False
    return True


def execute(job: dict) -> bool:
    status = read_status(job['name'])
    if status.get('state') == 'complete':
        return True
    environment = dict(
        os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
    )
    started = time.time()
    run = Path(job['run_dir'])
    pid = status.get('pid')
    if status.get('state') == 'training' and pid and process_alive(int(pid)):
        trained = wait_for_existing_training(job, int(pid))
    elif (run / 'checkpoints/final.pt').exists():
        trained = True
    else:
        trained = launch_training(job, environment)
    if not trained:
        write_status(job, 'training_failed')
        return False
    if not evaluate_and_render(job, environment):
        write_status(job, 'evaluation_failed')
        return False
    write_status(job, 'complete', wall_seconds=time.time() - started)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-workers', type=int, default=12)
    args = parser.parse_args()
    jobs = json.loads((ROOT / 'jobs.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        results = []
        for future in concurrent.futures.as_completed([pool.submit(execute, job) for job in jobs]):
            try:
                results.append(future.result())
            except Exception as error:
                print(repr(error), flush=True)
                results.append(False)
    (ROOT / ('ALL_COMPLETE' if all(results) else 'QUEUE_FINISHED_WITH_ERRORS')).touch()


if __name__ == '__main__':
    main()
