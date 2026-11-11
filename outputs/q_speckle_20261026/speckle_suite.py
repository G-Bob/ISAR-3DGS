"""Coherent-speckle robustness suite (2026-10-26) on the lambda_ent=0.3 main method. Reuses main_suite's queue (claims, deadline, stale reclaim)."""
import os, sys, json, threading, subprocess, time
from pathlib import Path
sys.path.insert(0, 'outputs/q_main_20261017')
import main_suite as ms
ms.ROOT = Path('outputs/q_speckle_20261026')
R = ms.R
def P(run, spec): run['perturb'] = spec; return run
D = lambda d, s: R(f'D{d:02d}_seed{s}', 1, 0, 0, 'sh3', seed=s, src=ms.DSRC, base=ms.DAILY / f'D{d:02d}', parent=f'daily D{d:02d}')
ms.RUNS = [
 P(R('S1_sh3_mask_speckle1', 1, 0, 0, 'sh3'), 'speckle:1'), P(R('S2_sh3_mask_speckle3', 1, 0, 0, 'sh3'), 'speckle:3'),
 P(R('S3_sh3_mask_speckle10', 1, 0, 0, 'sh3'), 'speckle:10'),
 P(R('S4_sh3_mask_speckleclip1', 1, 0, 0, 'sh3'), 'speckleclip:1'), P(R('S5_sh3_mask_speckleclip3', 1, 0, 0, 'sh3'), 'speckleclip:3'),
 P(R('S6_sh3_mask_speckleclip10', 1, 0, 0, 'sh3'), 'speckleclip:10'),
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
