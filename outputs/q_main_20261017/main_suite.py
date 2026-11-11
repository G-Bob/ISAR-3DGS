"""Main-method re-run (2026-10-17): every paper run of our method family with lambda_support_binary=0.3 (support binarisation).
Shared on-disk queue: a run is claimed by creating its directory, so several GPU jobs can cooperate. Usage: python main_suite.py [--workers 3] [--smoke NAME]"""
import argparse, json, os, subprocess, sys, threading, time, socket
from pathlib import Path
ROOT = Path('outputs/q_main_20261017')
RSRC = Path('outputs/echo_mask_robustness_20261009/source_snapshot')
DSRC = Path('outputs/sh3_echo_daily_recovery_20261011/source_snapshot')
BASE = Path('outputs/echo_mask_robustness_20261009/sh3_echo_seed43')
DAILY = Path('outputs/sh3_echo_daily_recovery_20261011')
DATA = '/rds/homes/g/gey/Yao/programs/GEIST-6/datasets/bsat'
LAM = dict(lambda_support_binary=0.3)
def R(name, mask, track, guard, hybrid='', seed=42, cfg=None, src=RSRC, base=BASE, evalmode=None, parent=''):
    return dict(name=name, mask=mask, track=track, guard=guard, hybrid=hybrid, seed=seed, cfg=dict(cfg or {}), src=str(src), base=str(base),
                evalmode=evalmode or ('sh3' if 'sh3' in hybrid else ('full_covariance' if 'full_covariance' in hybrid else 'no_visibility')), parent=parent)
RUNS = [
 R('F8_mlp_mask_track_guard', 1, 1, 1, parent='M19'), R('F5_mlp_mask', 1, 0, 0, parent='M18'), R('F1_mlp_none', 0, 0, 0, parent='M15'),
 R('S1_mlp_none_seed43', 0, 0, 0, seed=43), R('S2_mlp_none_seed44', 0, 0, 0, seed=44), R('S3_mlp_mask_seed43', 1, 0, 0, seed=43), R('S4_mlp_mask_seed44', 1, 0, 0, seed=44),
 R('F9_sh3_mask_track_guard', 1, 1, 1, 'sh3', parent='M30'), R('R1_mlp_aux_no_visibility', 1, 1, 1, cfg=dict(disable_visibility=True), parent='M31'),
 R('R2_mlp_aux_full_covariance', 1, 1, 1, 'full_covariance', parent='M32'),
 R('F2_mlp_track', 0, 1, 0, parent='M16'), R('F3_mlp_guard', 0, 0, 1, parent='M47'), R('F4_mlp_track_guard', 0, 1, 1, parent='M48'),
 R('F6_mlp_mask_track', 1, 1, 0, parent='M17'), R('F7_mlp_mask_guard', 1, 0, 1, parent='M20'),
 R('T1_mlp_aux_fixed10k', 1, 1, 1, cfg=dict(adaptive_interval=0), parent='Q41'), R('T2_mlp_track_fullimage_fixed10k', 0, 1, 0, cfg=dict(adaptive_interval=0), parent='Q44'),
 R('T3_mlp_aux_no_surface_fixed10k', 1, 1, 1, cfg=dict(adaptive_interval=0, lambda_surface=0.0), parent='Q45'),
] + [R(f'D{d:02d}_sh3_mask', 1, 0, 0, 'sh3', src=DSRC, base=DAILY / f'D{d:02d}', parent=f'daily D{d:02d}') for d in range(1, 17)]
def prepare(p, r):
    base = Path(p['base']); cfg = json.loads((base / 'config.json').read_text()); cfg.update(p['cfg']); cfg.update(LAM)
    cfg.update(seed=p['seed'], out_dir=str(r / 'training'), config=str(r / 'config.json'), encounter_names_file=str((base / 'train_cpis.txt').resolve()))
    (r / 'config.json').write_text(json.dumps(cfg, indent=2))
    (r / 'PLAN.json').write_text(json.dumps(dict(name=p['name'], seed=p['seed'], mode=p['evalmode'], parent=p['parent'], lambda_support_binary=0.3,
        budget='12k batch8 initial10k cap20k', evaluation='all and q030 (q050 computed by q_analysis)', status='q_main_20261017'), indent=2))
def env_for(p):
    return dict(os.environ, PYTHONPATH=str(Path(p['src']).resolve()) + ':.:outputs', GEN_SRC=p['src'], GEN_HYBRID=p['hybrid'],
                R6_MASK=str(p['mask']), R6_TRACK=str(p['track']), R6_GUARD=str(p['guard']), ROBUST_MODE='sh3_echo' if 'sh3' in p['hybrid'] else 'mlp')
def command(p, r, iters, names):
    return [sys.executable, '-u', str(ROOT / 'gen_worker.py'), '--config', str(r / 'config.json'), '--data_root', DATA, '--out_dir', str(r / 'training'),
            '--encounter_names_file', str(names), '--frames_per_encounter', '1', '--images_per_encounter', '1', '--iters', str(iters), '--views_per_batch', '8',
            '--checkpoint_interval', '500', '--preview_interval', '500', '--seed', str(p['seed']), '--device', 'cuda']
def execute(p, r):
    prepare(p, r); env = env_for(p)
    (r / 'started.json').write_text(json.dumps(dict(host=socket.gethostname(), job=os.environ.get('SLURM_JOB_ID'), time=time.time(), run=p), indent=2))
    with (r / 'train.log').open('w') as f:
        code = subprocess.call(command(p, r, 12000, Path(p['base']) / 'train_cpis.txt'), env=env, stdout=f, stderr=subprocess.STDOUT)
    (r / 'exit.json').write_text(json.dumps(dict(code=code)))
    if code != 0: return
    with (r / 'evaluation.log').open('w') as f:
        ec = subprocess.call([sys.executable, '-u', str(RSRC / 'evaluate_suite.py'), str(r), '--hybrid', p['evalmode']], env=env, stdout=f, stderr=subprocess.STDOUT)
    (r / 'evaluation_exit.json').write_text(json.dumps(dict(code=ec)))
MIN_LEFT = 110 * 60  # a run needs ~95 min incl. evaluation
def _secs(raw):
    raw = raw.strip()
    if not raw or raw in ('INVALID', 'UNLIMITED'): return 10 ** 9
    days = 0
    if '-' in raw: d, raw = raw.split('-'); days = int(d)
    x = [int(v) for v in raw.split(':')]
    while len(x) < 3: x.insert(0, 0)
    return days * 86400 + x[0] * 3600 + x[1] * 60 + x[2]
def remaining():
    j = os.environ.get('SLURM_JOB_ID')
    if not j: return 10 ** 9
    try: return _secs(subprocess.check_output(['squeue', '-j', j, '-h', '-o', '%L'], text=True))
    except Exception: return 10 ** 9
def alive(job):
    if not job: return False
    try: return bool(subprocess.check_output(['squeue', '-j', str(job), '-h', '-o', '%T'], text=True, stderr=subprocess.DEVNULL).strip())
    except Exception: return False
def stale(r):
    if (r / 'exit.json').exists(): return False
    c = r / 'claim.json'
    if not c.exists(): return time.time() - r.stat().st_mtime > 600
    return not alive(json.loads(c.read_text()).get('job'))
def claim():
    if remaining() < MIN_LEFT:
        print(time.strftime('%H:%M:%S'), 'deadline: no new claims', flush=True); return None, None
    for p in RUNS:
        r = ROOT / p['name']
        if r.exists():
            if not stale(r): continue
            try: os.rename(r, r.with_name(r.name + '.stale%d' % int(time.time())))
            except OSError: continue
        try: r.mkdir(exist_ok=False)
        except FileExistsError: continue
        (r / 'claim.json').write_text(json.dumps(dict(job=os.environ.get('SLURM_JOB_ID'), host=socket.gethostname(), time=time.time())))
        return p, r
    return None, None
def all_done():
    return all((ROOT / p['name'] / 'evaluation_exit.json').exists() for p in RUNS)
def loop(i):
    time.sleep(30 * i)
    while True:
        p, r = claim()
        if p is None: return
        print(time.strftime('%H:%M:%S'), 'worker', i, 'start', p['name'], flush=True)
        try: execute(p, r)
        except Exception as e: (r / 'crash.txt').write_text(repr(e))
        print(time.strftime('%H:%M:%S'), 'worker', i, 'done', p['name'], flush=True)
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=3); ap.add_argument('--smoke'); a = ap.parse_args()
    if a.smoke:
        p = [x for x in RUNS if x['name'] == a.smoke][0]; r = ROOT / 'smoke' / p['name']; r.mkdir(parents=True, exist_ok=False); prepare(p, r)
        code = subprocess.call(command(p, r, 2, BASE / 'smoke_train16.txt' if (BASE / 'smoke_train16.txt').exists() else Path('outputs/echo_mask_robustness_20261009/smoke_train16.txt')), env=env_for(p))
        print('SMOKE', p['name'], code); sys.exit(code)
    ts = [threading.Thread(target=loop, args=(i,)) for i in range(a.workers)]
    [t.start() for t in ts]; [t.join() for t in ts]; print('QUEUE_EMPTY', flush=True)
    if all_done():
        try:
            os.mkdir(ROOT / 'collect.lock')
            subprocess.call([sys.executable, '-u', str(ROOT / 'collect.py')], stdout=open(ROOT / 'collect.log', 'w'), stderr=subprocess.STDOUT)
            (ROOT / 'ALL_DONE').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))
        except FileExistsError: pass
