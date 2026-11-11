"""Controls re-parented on the proposed SH3+mask (lambda_ent=0.3), 2026-10-18. Reuses main_suite's queue (claims, deadline, stale reclaim)."""
import os, sys, json, threading, subprocess, time
from pathlib import Path
sys.path.insert(0, 'outputs/q_main_20261017')
import main_suite as ms
ms.ROOT = Path('outputs/q_ctrl_20261018')
R = ms.R
def P(run, spec): run['perturb'] = spec; return run
D = lambda d, s: R(f'D{d:02d}_seed{s}', 1, 0, 0, 'sh3', seed=s, src=ms.DSRC, base=ms.DAILY / f'D{d:02d}', parent=f'daily D{d:02d}')
ms.RUNS = [
 R('C6_sh3_fullimage', 0, 0, 0, 'sh3', parent='SH3 parent, no mask'),
 R('C1_sh3_mask_no_visibility', 1, 0, 0, 'sh3', cfg=dict(disable_visibility=True), parent='SH3+mask, M31-type'),
 R('C2_sh3_mask_full_covariance', 1, 0, 0, 'sh3,full_covariance', parent='SH3+mask, M32-type'),
 R('C3_sh3_mask_fixed10k', 1, 0, 0, 'sh3', cfg=dict(adaptive_interval=0), parent='SH3+mask, Q41-type'),
 R('C4_sh3_fullimage_fixed10k', 0, 0, 0, 'sh3', cfg=dict(adaptive_interval=0), parent='SH3, Q44-type'),
 R('C5_sh3_mask_no_surface_fixed10k', 1, 0, 0, 'sh3', cfg=dict(adaptive_interval=0, lambda_surface=0.0), parent='SH3+mask, Q45-type'),
 R('C7_sh3_mask_guard', 1, 0, 1, 'sh3', parent='SH3+mask + guard'),
 R('C8_sh3_mask_track', 1, 1, 0, 'sh3', parent='SH3+mask + tracking'),
]
_env = ms.env_for
def env_for(p):
    e = _env(p); e['GEN_PERTURB'] = p.get('perturb', ''); return e
ms.env_for = env_for
def all_done(): return all((ms.ROOT / p['name'] / 'evaluation_exit.json').exists() for p in ms.RUNS)
if __name__ == '__main__':
    ms.ROOT.mkdir(exist_ok=True)
    ts = [threading.Thread(target=ms.loop, args=(i,)) for i in range(3)]
    [t.start() for t in ts]; [t.join() for t in ts]; print('QUEUE_EMPTY', flush=True)
    if all_done():
        try:
            os.mkdir(ms.ROOT / 'collect.lock')
            subprocess.call([sys.executable, '-u', str(ms.ROOT / 'collect_robust.py')], stdout=open(ms.ROOT / 'collect.log', 'w'), stderr=subprocess.STDOUT)
            (ms.ROOT / 'ALL_DONE').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))
        except FileExistsError: pass
