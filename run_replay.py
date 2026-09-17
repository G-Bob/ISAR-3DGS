"""Run the two replay variants in parallel; each variant remains serial over K."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import time

BASE = Path("/root/autodl-fs/radar_gs_isar_enc")
ROOT = BASE / "outputs/radar_gs/skynet5d4_replay_v24_20260910"
DATA = Path("/root/Yao/datasets/skynet5d-4")
WORK = Path(__file__).resolve().parent
PYTHON = "/root/miniconda3/bin/python"

def set_status(root, job, state, **extra):
    path=root/"status"/f"{job['name']}.json";path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(dict(job,state=state,**extra),indent=2));tmp.replace(path)

def call(root, job, state, command, log, env):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as stream:
        p=subprocess.Popen(command,cwd=WORK,env=env,stdout=stream,stderr=subprocess.STDOUT)
        set_status(root,job,state,pid=p.pid,command=command,timestamp=time.time())
        return p.wait()==0

def evaluate(root,job,mode,env):
    out=root/("evaluation_current" if mode=="current" else "evaluation_cumulative")/job['name']
    if (out/'evaluation_metrics.json').exists(): return True
    run=Path(job['run_dir']); preset=job['current_preset'] if mode=='current' else job['cumulative_preset']
    cmd=[PYTHON,'-u','evaluate_ablation.py','--checkpoint',str(run/'checkpoints/final.pt'),'--data_root',str(DATA),'--run_dir',str(run),'--out_dir',str(out),'--device','cuda','--split','all','--evaluation_preset_json',preset,'--max_eval_views','240','--eval_seed','42','--preview_count','12','--q_thresholds','0.3','0.5','--no_export_fused_ply','--gt_geometry',job['gt']]
    return call(root,job,f'evaluating_{mode}',cmd,out/'evaluation.log',env)

def chain(variant,jobs):
    root=ROOT/variant;env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    for job in jobs:
        run=Path(job['run_dir']);final=run/'checkpoints/final.pt'
        if not final.exists():
            cmd=[PYTHON,'-u','run_reconstruction.py','--config',job['config'],'--data_root',str(DATA),'--out_dir',str(run)]
            if not call(root,job,'training',cmd,run/'train.log',env): set_status(root,job,'training_failed');return False
        if not evaluate(root,job,'current',env) or not evaluate(root,job,'cumulative',env): set_status(root,job,'evaluation_failed');return False
        out=root/'evaluation_cumulative'/job['name']
        if not (out/'pointcloud_native_cad.png').exists():
            if not call(root,job,'rendering',[PYTHON,'render_sequential_result.py','--run',str(run),'--out',str(out)],out/'render.log',env): set_status(root,job,'render_failed');return False
        set_status(root,job,'complete',finish=time.time())
    (root/'ALL_COMPLETE').touch();return True

def main():
    chains=json.loads((ROOT/'chains.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results={variant:future.result() for variant,future in [(v,pool.submit(chain,v,j)) for v,j in chains.items()]}
    (ROOT/('ALL_COMPLETE' if all(results.values()) else 'FINISHED_WITH_ERRORS')).touch()

if __name__=='__main__': main()
