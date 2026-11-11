"""q-support study (2026-10-16): SH3+Echo-mask, only support-prior weights changed. Same frozen source/pipeline as echo_mask_robustness_20261009."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('outputs/q_support_20261016'); SRC=Path('outputs/echo_mask_robustness_20261009/source_snapshot')
BASE=Path('outputs/echo_mask_robustness_20261009/sh3_echo_seed43')
DATA='/rds/homes/g/gey/Yao/programs/GEIST-6/datasets/bsat'
VARIANTS={'ent010':dict(lambda_support_binary=0.01),'ent030':dict(lambda_support_binary=0.03),'ent100':dict(lambda_support_binary=0.1),'ent300':dict(lambda_support_binary=0.3),'ent1000':dict(lambda_support_binary=1.0),'ent2000':dict(lambda_support_binary=2.0),'ent10000':dict(lambda_support_binary=10.0),
          'nopriors':dict(lambda_support_sparsity=0.0,lambda_support_binary=0.0,lambda_support_graph=0.0)}
def prepare(variant,seed):
    r=ROOT/f'{variant}_seed{seed}'; r.mkdir(exist_ok=False)
    cfg=json.loads((BASE/'config.json').read_text()); cfg.update(VARIANTS[variant]); cfg.update(seed=seed,out_dir=str(r/'training'),config=str(r/'config.json'),encounter_names_file=str((BASE/'train_cpis.txt').resolve()))
    (r/'config.json').write_text(json.dumps(cfg,indent=2))
    plan=json.loads((BASE/'PLAN.json').read_text()); plan.update(name=r.name,seed=seed,status='q_support_20261016 support-prior variant '+variant); (r/'PLAN.json').write_text(json.dumps(plan,indent=2)); return r
def command(r,seed):
    return [sys.executable,'-u',str(SRC/'robust_worker.py'),'--config',str(r/'config.json'),'--data_root',DATA,'--out_dir',str(r/'training'),'--encounter_names_file',str(BASE/'train_cpis.txt'),'--frames_per_encounter','1','--images_per_encounter','1','--iters','12000','--views_per_batch','8','--checkpoint_interval','500','--preview_interval','500','--seed',str(seed),'--device','cuda']
if __name__=='__main__':
    jobs=[a.split(':') for a in sys.argv[1:]]
    env=dict(os.environ,PYTHONPATH=str(SRC.resolve())+':.:outputs',ROBUST_MODE='sh3_echo')
    procs=[]
    for v,s in jobs:
        r=prepare(v,int(s)); f=(r/'train.log').open('w')
        procs.append((r,subprocess.Popen(command(r,int(s)),env=env,stdout=f,stderr=subprocess.STDOUT))); time.sleep(20)
    for r,p in procs:
        code=p.wait(); (r/'exit.json').write_text(json.dumps(dict(code=code)))
        if code==0:
            with (r/'evaluation.log').open('w') as f: ec=subprocess.call([sys.executable,'-u',str(SRC/'evaluate_suite.py'),str(r),'--hybrid','sh3'],env=env,stdout=f,stderr=subprocess.STDOUT)
            (r/'evaluation_exit.json').write_text(json.dumps(dict(code=ec)))
    print('WAVE_DONE',flush=True)
