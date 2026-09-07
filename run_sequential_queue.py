"""Persistent bounded queue: training then native-CAD evaluation and rendering."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/root/autodl-fs/radar_gs_isar_enc/outputs/radar_gs/skynet5d3_sequential_v21_20260907')
WORK=Path(__file__).resolve().parent
PY='/root/miniconda3/bin/python'

def write(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2));temp.replace(path)

def execute(job):
    name=job['name'];run=Path(job['run_dir']);run.mkdir(parents=True,exist_ok=True)
    status=ROOT/'status'/f'{name}.json'
    if (run/'train.log').exists():
        raise RuntimeError(f'Refuse duplicate start: {name}')
    start=time.time()
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    cmd=[PY,'-u','run_reconstruction.py','--config',job['config'],
         '--data_root',f'/root/Yao/datasets/skynet5d-3/{job["scenario"]}', '--out_dir',str(run)]
    with (run/'train.log').open('x') as f:
        p=subprocess.Popen(cmd,cwd=WORK,env=env,stdout=f,stderr=subprocess.STDOUT)
        write(status,dict(job,state='training',pid=p.pid,start=start,command=cmd))
        code=p.wait()
    if code:
        write(status,dict(job,state='training_failed',returncode=code));return False
    out=ROOT/'evaluation'/name;out.mkdir(parents=True,exist_ok=True)
    cmd=[PY,'-u','evaluate_ablation.py','--checkpoint',str(run/'checkpoints/final.pt'),'--data_root',f'/root/Yao/datasets/skynet5d-3/{job["scenario"]}','--run_dir',str(run),'--out_dir',str(out),'--device','cuda','--split','all','--max_eval_views','240','--eval_seed','42','--preview_count','12','--q_thresholds','0.3','0.5','--no_export_fused_ply','--gt_geometry',job['gt']]
    with (out/'evaluation.log').open('x') as f:
        p=subprocess.Popen(cmd,cwd=WORK,env=env,stdout=f,stderr=subprocess.STDOUT)
        write(status,dict(job,state='evaluating_in_sample',pid=p.pid,command=cmd))
        code=p.wait()
    if code:
        write(status,dict(job,state='evaluation_failed',returncode=code));return False
    with (out/'render.log').open('x') as f:
        code=subprocess.call([PY,'render_sequential_result.py','--run',str(run),'--out',str(out)],cwd=WORK,env=env,stdout=f,stderr=subprocess.STDOUT)
    write(status,dict(job,state='complete' if not code else 'render_failed',wall_seconds=time.time()-start))
    return code==0

if __name__=='__main__':
    os.chdir(WORK)
    lock=ROOT/'QUEUE_SUBMITTED'
    with lock.open('x') as f:f.write(str(os.getpid()))
    jobs=json.loads((ROOT/'jobs.json').read_text())
    (ROOT/'status').mkdir(exist_ok=True)
    for job in jobs:write(ROOT/'status'/f'{job["name"]}.json',dict(job,state='queued'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(execute,job) for job in jobs]
        results=[]
        for f in concurrent.futures.as_completed(futures):
            try:results.append(f.result())
            except Exception as e:print(repr(e),flush=True);results.append(False)
    (ROOT/('ALL_COMPLETE' if all(results) else 'QUEUE_FINISHED_WITH_ERRORS')).touch()
