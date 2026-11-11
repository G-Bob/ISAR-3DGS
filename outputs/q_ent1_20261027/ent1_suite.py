"""lambda_ent = 1 re-run of every Skynet paper run of the proposed method (2026-10-27). main_suite.py here = q_main's with LAM = 1.0."""
import os, sys, subprocess, threading, time
from pathlib import Path
sys.path.insert(0, 'outputs/q_ent1_20261027')
import main_suite as ms
assert ms.LAM['lambda_support_binary'] == 1.0 and str(ms.ROOT) == 'outputs/q_ent1_20261027'
R = ms.R
def P(run, spec): run['perturb'] = spec; return run
daily = [p for p in ms.RUNS if p['name'].startswith('D')]
assert len(daily) == 16
ms.RUNS = [
 R('C6_sh3_fullimage', 0, 0, 0, 'sh3'), R('C1_sh3_mask_no_visibility', 1, 0, 0, 'sh3', cfg=dict(disable_visibility=True)),
 R('C2_sh3_mask_full_covariance', 1, 0, 0, 'sh3,full_covariance'), R('C3_sh3_mask_fixed10k', 1, 0, 0, 'sh3', cfg=dict(adaptive_interval=0)),
 R('C4_sh3_fullimage_fixed10k', 0, 0, 0, 'sh3', cfg=dict(adaptive_interval=0)), R('C5_sh3_mask_no_surface_fixed10k', 1, 0, 0, 'sh3', cfg=dict(adaptive_interval=0, lambda_surface=0.0)),
 P(R('N1_sh3_mask_snr30', 1, 0, 0, 'sh3'), 'rician:30'), P(R('N2_sh3_mask_snr20', 1, 0, 0, 'sh3'), 'rician:20'), P(R('N7_sh3_mask_snr10', 1, 0, 0, 'sh3'), 'rician:10'),
 P(R('L1_sh3_mask_los0p25', 1, 0, 0, 'sh3'), 'los:0.25'), P(R('L5_sh3_mask_los0p5', 1, 0, 0, 'sh3'), 'los:0.5'), P(R('L2_sh3_mask_los1p0', 1, 0, 0, 'sh3'), 'los:1.0'),
 P(R('L6_sh3_mask_los2p0', 1, 0, 0, 'sh3'), 'los:2.0'),
 P(R('S6_sh3_mask_speckleclip10', 1, 0, 0, 'sh3'), 'speckleclip:10'), P(R('S5_sh3_mask_speckleclip3', 1, 0, 0, 'sh3'), 'speckleclip:3'),
 P(R('S4_sh3_mask_speckleclip1', 1, 0, 0, 'sh3'), 'speckleclip:1'),
] + daily
_env = ms.env_for
def env_for(p):
    e = _env(p); e['GEN_PERTURB'] = p.get('perturb', ''); return e
ms.env_for = env_for
if __name__ == '__main__':
    ms.ROOT.mkdir(exist_ok=True)
    ts = [threading.Thread(target=ms.loop, args=(i,)) for i in range(int(sys.argv[1]) if len(sys.argv) > 1 else 3)]
    [t.start() for t in ts]; [t.join() for t in ts]; print('QUEUE_EMPTY', flush=True)
    if all((ms.ROOT / p['name'] / 'evaluation_exit.json').exists() for p in ms.RUNS):
        try:
            os.mkdir(ms.ROOT / 'collect.lock')
            subprocess.call([sys.executable, '-u', str(ms.ROOT / 'collect_robust.py')], stdout=open(ms.ROOT / 'collect.log', 'w'), stderr=subprocess.STDOUT)
            (ms.ROOT / 'ALL_DONE').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))
        except FileExistsError: pass
