import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, 'outputs/q_main_20261017')
from plyread import read_ply
R = Path('outputs/baseline_robust_20261026')
GT = np.load('outputs/bsat_observation_days_20261004/skynet_gt_blender_5000.npy'); TG = cKDTree(GT)
K = dict(snr30=11422, snr20=11352, snr10=11587, spk10=11253, spk3=11089, spk1=10639, los0p25=11460, los0p5=11457, los1p0=11459, los2p0=11432)
def met(P):
    a = TG.query(P)[0]; b = cKDTree(P).query(GT)[0]
    return dict(cd1=0.5*(a.mean()+b.mean()), cd2=0.5*((a**2).mean()+(b**2).mean()), hd95=max(np.percentile(a,95), np.percentile(b,95)), acc=a.mean(), comp=b.mean(), cov=100*(b<0.1).mean())
out = {}
for d in sorted(R.iterdir()):
    if not d.is_dir() or '.' in d.name or not (d / 'evaluation_exit.json').exists(): continue
    m, c = d.name.rsplit('_', 1)
    ply = (d / 'run' if m == 'pei' else d) / 'gaussians_final.ply'
    v = read_ply(ply); xyz = np.stack([v['x'], v['y'], v['z']], 1).astype(float); s = np.asarray(v['target_support_probability'], float)
    top = np.argsort(-s)[:K[c]]
    ev = next(iter(sorted((d / 'run' if m == 'pei' else d).glob('evaluation_common475/summary.json'))), None)
    ps = {}
    if ev:
        j = json.loads(ev.read_text()); ps = {k: v for k, v in j.items() if isinstance(v, (int, float)) and ('psnr' in k.lower() or 'ssim' in k.lower())}
    out[d.name] = dict(method=m, cond=c, K=K[c], N_all=len(xyz), topK=met(xyz[top]), all=met(xyz), render=ps)
    print(d.name, len(xyz), round(out[d.name]['topK']['cd1'], 4), round(out[d.name]['topK']['cov'], 1), ps, flush=True)
(R / 'collected.json').write_text(json.dumps(out, indent=1))