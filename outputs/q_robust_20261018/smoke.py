import sys, subprocess
from pathlib import Path
sys.argv = sys.argv[:1] + sys.argv[1:]
sys.path.insert(0, 'outputs/q_robust_20261018')
import robust_suite as rs
ms = rs.ms
for name in sys.argv[1:]:
    p = [x for x in ms.RUNS if x['name'] == name][0]; r = ms.ROOT / 'smoke' / name; r.mkdir(parents=True, exist_ok=True); ms.prepare(p, r)
    code = subprocess.call(ms.command(p, r, 2, Path('outputs/echo_mask_robustness_20261009/smoke_train16.txt')), env=ms.env_for(p), stdout=open(r / 'smoke.log', 'w'), stderr=subprocess.STDOUT)
    log = (r / 'smoke.log').read_text()
    print('SMOKE', name, code, [l for l in log.splitlines() if 'GEN_PERTURB' in l or 'R6_READY' in l or 'Error' in l][:3], flush=True)
