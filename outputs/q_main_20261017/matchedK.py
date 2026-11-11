import numpy as np, json
from pathlib import Path
from plyread import read_ply
from scipy.spatial import cKDTree
GT = np.load('outputs/bsat_observation_days_20261004/skynet_gt_blender_5000.npy'); TG = cKDTree(GT)
P = Path('outputs/paper_reproduction'); K = 11472
runs = {'Pei uniform': 'P30_Pei_uniform_wide_restart', 'Pei training': 'P31_Pei_trainpoints_wide_restart',
        'ISAR-GS': 'P32_ISARGS_complex_continuum', 'RNSplat base': 'P33_RNS_alpha_base', 'RNSplat gain': 'P34_RNS_alpha_RSU'}
def metrics(p):
    dp = TG.query(p)[0]; dg = cKDTree(p).query(GT)[0]
    return dict(N=len(p), cd1=0.5*(dp.mean()+dg.mean()), cd2=0.5*((dp**2).mean()+(dg**2).mean()),
                hd95=max(np.percentile(dp, 95), np.percentile(dg, 95)), acc=dp.mean(), comp=dg.mean(), cov=100*(dg < 0.1).mean())
out = {}
for name, r in runs.items():
    v = read_ply(P / r / "gaussians_final.ply")
    xyz = np.stack([v['x'], v['y'], v['z']], 1).astype(float); a = np.asarray(v['target_support_probability'], float)
    top = np.argsort(-a)[:K]
    rng = np.random.default_rng(0); rnd = rng.choice(len(xyz), K, replace=False)
    out[name] = {'all': metrics(xyz), 'a004': metrics(xyz[a >= 0.04]), 'topK': metrics(xyz[top]), 'randK': metrics(xyz[rnd]),
                 'alpha_at_K': float(a[top[-1]])}
    for k, m in out[name].items():
        print(name, k, m if isinstance(m, float) else ' '.join('%s %.5g' % kv for kv in m.items()))
json.dump(out, open('outputs/q_main_20261017/matchedK_baselines.json', 'w'), indent=1)
