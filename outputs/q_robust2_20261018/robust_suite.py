"""Phase 2: noise breakdown, robustness seeds, LOS curve, more daily seeds (2026-10-18) on the lambda_ent=0.3 main method. Reuses main_suite's queue (claims, deadline, stale reclaim)."""
import os, sys, json, threading, subprocess, time
from pathlib import Path
sys.path.insert(0, 'outputs/q_main_20261017')
import main_suite as ms
ms.ROOT = Path('outputs/q_robust2_20261018')
R = ms.R
def P(run, spec): run['perturb'] = spec; return run
D = lambda d, s: R(f'D{d:02d}_seed{s}', 1, 0, 0, 'sh3', seed=s, src=ms.DSRC, base=ms.DAILY / f'D{d:02d}', parent=f'daily D{d:02d}')
ms.RUNS = [
 P(R('N7_sh3_mask_snr10', 1, 0, 0, 'sh3'), 'rician:10'), P(R('N8_mlp_mask_snr10', 1, 0, 0), 'rician:10'), P(R('N9_mlp_none_snr10', 0, 0, 0), 'rician:10'),
 P(R('N2s43_sh3_mask_snr20', 1, 0, 0, 'sh3', seed=43), 'rician:20'), P(R('N2s44_sh3_mask_snr20', 1, 0, 0, 'sh3', seed=44), 'rician:20'),
 P(R('N6s43_mlp_mask_snr20', 1, 0, 0, seed=43), 'rician:20'), P(R('N6s44_mlp_mask_snr20', 1, 0, 0, seed=44), 'rician:20'),
 P(R('N4s43_mlp_none_snr20', 0, 0, 0, seed=43), 'rician:20'), P(R('N4s44_mlp_none_snr20', 0, 0, 0, seed=44), 'rician:20'),
 P(R('L5_sh3_mask_los0p5', 1, 0, 0, 'sh3'), 'los:0.5'), P(R('L6_sh3_mask_los2p0', 1, 0, 0, 'sh3'), 'los:2.0'), P(R('L7_mlp_mask_los0p25', 1, 0, 0), 'los:0.25'),
 D(10, 43), D(10, 44), D(6, 43), D(6, 44), D(2, 43), D(2, 44), D(14, 43), D(14, 44),
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
