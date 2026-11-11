"""lambda_ent = 1: proposed q>=0.5 / all-centre metrics and K-matched baselines (2026-10-28). Run from ISAR-3DGS root."""
import json, sys
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
sys.path.insert(0, 'outputs/q_main_20261017')
from plyread import read_ply
O = Path('outputs')

def load(ply):
    v = read_ply(ply)
    return np.stack([v['x'], v['y'], v['z']], 1).astype(float), np.asarray(v['target_support_probability'], float)

def metrics(p, GT, TG):
    dp = TG.query(p)[0]; dg = cKDTree(p).query(GT)[0]
    return dict(N=len(p), cd1=0.5 * (dp.mean() + dg.mean()), cd2=0.5 * ((dp ** 2).mean() + (dg ** 2).mean()),
                hd95=max(np.percentile(dp, 95), np.percentile(dg, 95)), acc=dp.mean(), comp=dg.mean(), cov=100 * (dg < 0.1).mean())

def topk(ply, K, GT, TG):
    x, s = load(ply)
    return metrics(x[np.argsort(-s)[:K]], GT, TG)

out = {}
for tgt, gt, prop, base in [
    ('skynet', 'bsat_observation_days_20261004/skynet_gt_blender_5000.npy',
     [O / f'q_support_20261016/ent1000_seed{s}/training/gaussians_final.ply' for s in (42, 43, 44)],
     {'pei': O / 'paper_reproduction/P30_Pei_uniform_wide_restart/gaussians_final.ply',
      'isargs': O / 'paper_reproduction/P32_ISARGS_complex_continuum/gaussians_final.ply',
      'rns': O / 'paper_reproduction/P33_RNS_alpha_base/gaussians_final.ply'}),
    ('goes', 'goes_main_20261025/goes_gt_blender_5000.npy',
     [O / 'goes_ent1_20261027/P_sh3_mask/training/gaussians_final.ply'],
     {'pei': O / 'goes_main_20261025/B_Pei_uniform/run/gaussians_final.ply',
      'isargs': O / 'goes_main_20261025/B_ISARGS/gaussians_final.ply',
      'rns': O / 'goes_main_20261025/B_RNSplat_base/gaussians_final.ply'})]:
    GT = np.load(O / gt); TG = cKDTree(GT); r = out[tgt] = {}
    for i, ply in enumerate(prop):
        x, q = load(ply)
        r[f'prop{i}'] = metrics(x[q >= 0.5], GT, TG); r[f'prop{i}_all'] = metrics(x, GT, TG)
        r[f'prop{i}']['mid'] = 100 * ((q > 0.1) & (q < 0.9)).mean()
    K = r['prop0']['N']
    for m, ply in base.items():
        r[m] = topk(ply, K, GT, TG)
    print(tgt, K, {k: round(v['cd1'], 5) for k, v in r.items()}, flush=True)

# baseline robustness: K = proposed N (lambda_ent = 1) under the same condition
GT = np.load(O / 'bsat_observation_days_20261004/skynet_gt_blender_5000.npy'); TG = cKDTree(GT)
cond = dict(snr30='N1_sh3_mask_snr30', snr20='N2_sh3_mask_snr20', snr10='N7_sh3_mask_snr10',
            spk10='S6_sh3_mask_speckleclip10', spk3='S5_sh3_mask_speckleclip3', spk1='S4_sh3_mask_speckleclip1',
            los0p25='L1_sh3_mask_los0p25', los0p5='L5_sh3_mask_los0p5', los1p0='L2_sh3_mask_los1p0', los2p0='L6_sh3_mask_los2p0')
B = O / 'baseline_robust_20261026'; rb = out['robust'] = {}
for c, run in cond.items():
    x, q = load(O / 'q_ent1_20261027' / run / 'training/gaussians_final.ply'); K = int((q >= 0.5).sum())
    rb[c] = {'K': K}
    for m in ('pei', 'isargs', 'rns_base'):
        d = B / f'{m}_{c}'; ply = (d / 'run' if m == 'pei' else d) / 'gaussians_final.ply'
        rb[c][m] = topk(ply, K, GT, TG)['cd1'] if ply.exists() else None
    print(c, rb[c], flush=True)
json.dump(out, open(O / 'q_ent1_20261027/matchedK_ent1.json', 'w'), indent=1)
print('WROTE')
