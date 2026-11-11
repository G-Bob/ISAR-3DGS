"""Collect LOS 5/10 deg (2026-10-28): SC-GS q>=0.5 metrics + test PSNR/SSIM, baselines' top-K CD-L1 at the same K."""
import json, sys
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
sys.path.insert(0, 'outputs/q_main_20261017')
from plyread import read_ply
O = Path('outputs'); GT = np.load(O / 'bsat_observation_days_20261004/skynet_gt_blender_5000.npy'); TG = cKDTree(GT)

def load(ply):
    v = read_ply(ply)
    return np.stack([v['x'], v['y'], v['z']], 1).astype(float), np.asarray(v['target_support_probability'], float)

def metrics(p):
    dp = TG.query(p)[0]; dg = cKDTree(p).query(GT)[0]
    return dict(N=len(p), cd1=0.5 * (dp.mean() + dg.mean()), cd2=0.5 * ((dp ** 2).mean() + (dg ** 2).mean()),
                hd95=max(np.percentile(dp, 95), np.percentile(dg, 95)), cov=100 * (dg < 0.1).mean())

out = {}
for c, run in (('los5p0', 'L7_sh3_mask_los5p0'), ('los10p0', 'L8_sh3_mask_los10p0')):
    d = O / 'q_ent1_20261027' / run; x, q = load(d / 'training/gaussians_final.ply')
    r = metrics(x[q >= 0.5]); r['all_cd1'] = metrics(x)['cd1']; r['mid'] = 100 * ((q > 0.1) & (q < 0.9)).mean()
    s = json.load(open(d / 'evaluation_common475/summary.json'))['image']; r['psnr'] = s['mean_psnr_db']; r['ssim'] = s['mean_ssim']
    for m in ('pei', 'isargs', 'rns_base'):
        b = O / 'baseline_robust_20261026' / f'{m}_{c}'; ply = (b / 'run' if m == 'pei' else b) / 'gaussians_final.ply'
        if ply.exists():
            xb, sb = load(ply); r[m] = metrics(xb[np.argsort(-sb)[:r['N']]])['cd1']; r[m + '_n'] = len(xb)
        else:
            r[m] = None
    out[c] = r; print(c, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
json.dump(out, open(O / 'q_ent1_20261027/los_big_collected.json', 'w'), indent=1)
