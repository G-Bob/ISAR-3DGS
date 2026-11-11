"""Finite registered suite. Prepare now; launch only with verified resource time."""
import argparse,concurrent.futures,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path('outputs/paper_reproduction');SOURCE=ROOT/'suite_source_snapshot';OLD=Path('outputs/bsat_methods8_20261004/M19_track_mask_guard_fast_restart')
GROUPS=[
 ('M30_M19_SH3','hybrid','sh3','M19','Replace neural scattering with bounded degree3 SH; same16 feature slots. Parameter family changes, original geometry/loss/track retained.'),
 ('M31_M19_no_visibility','hybrid','no_visibility','M19','Remove attenuation from image signal only; auxiliary visibility/evidence remain as M19. This isolates image attenuation, not full Pei renderer.'),
 ('M32_M19_full_covariance','hybrid','full_covariance','M19','Independent quaternion covariance, initialized to identical covariance. Surface normal prior remains a separate field; full covariance orientation decouples from it. Clone/prune/relocation copy quaternion rows.'),
 ('P35_Pei_mask_guard','paper','pei_mask','P31','Pei full covariance SH and additive rendering, replace image objective with M19 echo mask/Charbonnier/log/SSIM/backgroundguard policy. No tracking. Loss-policy combination, not pure mask-only effect.'),
 ('P32_ISARGS_complex_continuum','paper','isargs','P31','ComplexSH3 and complex RD sum then magnitude. Eq13 uses whole LOS-plane continuum integral pi*sqrt(detCov)*(1-exp(-9)); sorted front-to-back integrated-global attenuation, not ray-local occlusion. Imag coefficients normal0,.01 seed4232. Shape/SOR omitted; no official/native claim.'),
 ('P33_RNS_alpha_base','paper','rns_base','Pei infrastructure','Known geometry RGB alpha compositor, no PHARE. Frozen invertible scalar embedding [x,x²,sqrt(x+1e-6)-.001], scalar metrics channel0. Full covariance, RGB SH3, 3sigma capalpha.99, RGB.8L1+.2DSSIM; SHrest1e-4 and scale anisotropy1e-3 regularizers.'),
 ('P34_RNS_alpha_RSU','paper','rns_rsu','P33','Same known-geometry RNS adaptation plus shared channel gain 1+.25sigmoid(beta)*sinc(b*c)*exp(-.5(b*c)^2), c=[-1,0,1], b=softplus(band), initialband1 beta-2. Applied to features before alpha; no per-Gaussian/postrender PSF. Gain can still be absorbed into RGBSH, optimization effect only; not fullPHARE.'),
]


def prepare():
    assert not (ROOT/'suite_manifest.json').exists(),'Do not silently rewrite frozen registered suite'
    SOURCE.mkdir(exist_ok=False)
    shutil.copytree('radar_gs',SOURCE/'radar_gs',ignore=shutil.ignore_patterns('__pycache__','*.so','build'))
    shutil.copy2('run_reconstruction.py',SOURCE/'run_reconstruction.py')
    for name in ['hybrid_worker.py','alpha_cuda.py','adapted_papers.py','check_adaptations.py','check_hybrid.py','evaluate_suite.py','prepare_and_launch_suite.py','pei_bsat_wide.py','pei_linear_cuda_wide.py']:
        shutil.copy2(ROOT/name,SOURCE/name)
    shutil.copy2('outputs/round6_fast_worker.py',SOURCE/'round6_fast_worker.py')
    shutil.copy2('evaluate_ablation.py',SOURCE/'evaluate_ablation.py')
    for f in ROOT.glob('check_*.json'):shutil.copy2(f,SOURCE/f.name)
    (SOURCE/'sha256.json').write_text(json.dumps({str(f.relative_to(SOURCE)):hashlib.sha256(f.read_bytes()).hexdigest() for f in SOURCE.rglob('*') if f.is_file()},indent=2))
    base=json.loads((OLD/'config.json').read_text());pei=json.loads((ROOT/'P31_Pei_trainpoints_wide_restart/run_config.json').read_text())
    plans=[]
    for name,kind,mode,parent,details in GROUPS:
        run=ROOT/name;run.mkdir(exist_ok=False)
        cfg=dict(base if kind=='hybrid' else pei);cfg.update(out_dir=str(run/'training' if kind=='hybrid' else run),init_path=str(Path('outputs/bsat_methods8_20261004/M15_multicpi_tracks/initializer.npz').resolve()),config=str(run/'config.json'))
        cfg['encounter_names_file']=str((run/'train_cpis.txt').resolve());shutil.copy2(OLD/'train_cpis.txt',run/'train_cpis.txt')
        if mode=='no_visibility':cfg['disable_visibility']=True
        (run/'config.json').write_text(json.dumps(cfg,indent=2))
        plan={'name':name,'kind':kind,'mode':mode,'parent':parent,'hypothesis_and_assumptions':details,'budget':'1903train475test seed42 12k batch8 exactM15initial10k cap20k','selection':'all plus q030; paper opacity alsoalpha004; fixed GT/noICP; no heldout tuning','success':'Primary all-point CD-L1 and CD-L2 both >=5% lower than matched parent; additionally compare M19 all(.1304885623m/.0903821249m2) and fixedq030(.0559855214m/.01193145365m2), never mix selections. HD95 nonworse and coverage0.1m loss<=2pp. Report directional means/points, PSNRSSIM diagnostic','cost':'reused M15 init23.6008s separate; training sharedGPU reported; new initialization/probe costs separate','status':'registered_waiting_resource','resume':'No reliable resume; preserve incomplete directory and diagnose; never claim equivalent continuation','source_snapshot':str(SOURCE)}
        (run/'PLAN.json').write_text(json.dumps(plan,indent=2));plans.append(plan)
    (ROOT/'suite_manifest.json').write_text(json.dumps(plans,indent=2));print('SUITE_REGISTERED',len(plans),flush=True)


def remaining_seconds(job):
    raw=subprocess.check_output(['squeue','-j',str(job),'-h','-o','%L'],text=True).strip()
    if not raw:return 0
    days=0
    if '-' in raw:d,raw=raw.split('-');days=int(d)
    x=list(map(int,raw.split(':')));return days*86400+(x[0]*3600+x[1]*60+x[2] if len(x)==3 else x[0]*60+x[1])


def execute(plan):
    run=ROOT/plan['name'];kind=plan['kind'];mode=plan['mode']
    assert not (run/'started.json').exists(),'Registered job already started'
    check=ROOT/('check_hybrid_'+mode+'.json' if kind=='hybrid' else 'check_'+mode+'.json');assert json.loads(check.read_text())['passed']
    env=dict(os.environ,PYTHONPATH='.:outputs:'+str(SOURCE.resolve()),R6_MASK='1',R6_TRACK='1',R6_GUARD='1')
    if kind=='hybrid':
        env['HYBRID_MODE']=mode
        cmd=[sys.executable,'-u',str(SOURCE/'hybrid_worker.py'),'--config',str(run/'config.json'),'--data_root','/rds/homes/g/gey/Yao/programs/GEIST-6/datasets/bsat','--out_dir',str(run/'training'),'--encounter_names_file',str(run/'train_cpis.txt'),'--frames_per_encounter','1','--images_per_encounter','1','--iters','12000','--views_per_batch','8','--checkpoint_interval','500','--preview_interval','500','--seed','42','--device','cuda']
    else:
        env['PAPER_MODE']=mode;cmd=[sys.executable,'-u',str(SOURCE/'adapted_papers.py'),'--config',str(run/'config.json'),'--out',str(run/'training_stage'),'--init','train_points','--iters','12000']
        # Trainer requires nonexisting output; use isolated training_stage,
        # then expose artifacts in registered run while retaining that directory.
    with (run/'train.log').open('w') as log:
        process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        (run/'started.json').write_text(json.dumps({'pid':process.pid,'command':cmd,'time':time.time()},indent=2));code=process.wait()
    (run/'exit.json').write_text(json.dumps({'code':code}));print(plan['name'],'TRAIN_EXIT',code,flush=True)
    if code:return
    training=run/'training' if kind=='hybrid' else run/'training_stage'
    if kind=='paper':
        for f in training.iterdir():
            dest=run/f.name
            if not dest.exists():dest.symlink_to(f.resolve(),target_is_directory=f.is_dir())
    import torch
    ck=torch.load(training/'checkpoints/final.pt',map_location='cpu',weights_only=False)
    assert ck['iteration']==12000 and all(torch.isfinite(v).all() for v in ck['model'].values())
    assert any(training.glob('*history*.json')) and any(training.glob('*timing*.json'))
    (run/'completion_verified.json').write_text(json.dumps({'exit':code,'iteration':12000,'finite':True,'history_timing':True}))
    env['PAPER_MODE']=mode;ev=[sys.executable,'-u',str(SOURCE/'evaluate_suite.py'),str(run),'--hybrid' if kind=='hybrid' else '--paper',mode]
    with (run/'evaluation.log').open('w') as log:ec=subprocess.call(ev,env=env,stdout=log,stderr=subprocess.STDOUT)
    (run/'evaluation_exit.json').write_text(json.dumps({'code':ec}));print(plan['name'],'EVAL_EXIT',ec,flush=True)


def launch(job):
    hashes=json.loads((SOURCE/'sha256.json').read_text())
    assert all(hashlib.sha256((SOURCE/n).read_bytes()).hexdigest()==h for n,h in hashes.items()),'Frozen snapshot changed'
    assert json.loads((ROOT/'check_physical_assumptions.json').read_text())['passed']
    plans=json.loads((ROOT/'suite_manifest.json').read_text())
    for start in range(0,len(plans),2):
        current=[p for p in plans[start:start+2] if not (ROOT/p['name']/'started.json').exists()]
        if not current:continue
        # RNS stream cost can be hours; use conservative actual20k throughput.
        required=max(5400,max(int(json.loads((ROOT/('check_hybrid_'+p['mode']+'.json' if p['kind']=='hybrid' else 'check_'+p['mode']+'.json')).read_text()).get('stress20k_seconds',.08)*8*12000*1.5+1800) for p in current))
        if remaining_seconds(job)<required:
            print('WAIT_NEW_ALLOCATION required_seconds',required,'job',job,flush=True);return
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(execute,current))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--launch',action='store_true');parser.add_argument('--job',type=int);args=parser.parse_args()
    if args.prepare:prepare()
    if args.launch:assert args.job;launch(args.job)
