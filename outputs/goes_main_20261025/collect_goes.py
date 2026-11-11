"""Collect GOES results: Table I (matched point count, q>=0.5 vs top-K), Table II (all centres + PSNR/SSIM), D01-D16 curve."""
import json, sys
from pathlib import Path
import numpy as np, torch
from scipy.spatial import cKDTree
sys.path.insert(0, 'outputs/q_main_20261017')
from plyread import read_ply
ROOT = Path('outputs/goes_main_20261025')
GT = np.load(ROOT / 'goes_gt_blender_5000.npy'); TG = cKDTree(GT)


def metrics(p):
    p = np.asarray(p, float)
    if len(p) == 0: return dict(N=0)
    dp = TG.query(p)[0]; dg = cKDTree(p).query(GT)[0]
    return dict(N=int(len(p)), cd_l1=float(.5 * (dp.mean() + dg.mean())), cd_l2=float(.5 * ((dp ** 2).mean() + (dg ** 2).mean())),
                hd95=float(max(np.percentile(dp, 95), np.percentile(dg, 95))), acc=float(dp.mean()), comp=float(dg.mean()), cov01=float(100 * (dg <= .1).mean()))


def image(summary):
    if not summary.exists(): return {}
    s = json.loads(summary.read_text())['image']; return dict(psnr=s['mean_psnr_db'], ssim=s['mean_ssim'])


def gs(run):
    ck = torch.load(run / 'training/checkpoints/final.pt', map_location='cpu', weights_only=False)
    mu = ck['model']['means'].numpy(); q = torch.sigmoid(ck['model']['support_logits'] / ck['arguments'].get('final_support_temperature', .35)).numpy()
    return dict(q050=metrics(mu[q >= .5]), all=metrics(mu), mid=float(((q > .1) & (q < .9)).mean()), **image(run / 'evaluation_common475/summary.json'))


out = {}
P = ROOT / 'P_sh3_mask'
if (P / 'evaluation_exit.json').exists(): out['P_sh3_mask'] = gs(P)
K = out.get('P_sh3_mask', {}).get('q050', {}).get('N', 11472)
for name, sub in [('B_Pei_uniform', 'run'), ('B_ISARGS', ''), ('B_RNSplat_base', '')]:
    r = ROOT / name / sub if sub else ROOT / name
    if not (r / 'gaussians_final.ply').exists(): continue
    v = read_ply(r / 'gaussians_final.ply'); xyz = np.stack([v['x'], v['y'], v['z']], 1).astype(float); a = np.asarray(v['target_support_probability'], float)
    out[name] = dict(topK=metrics(xyz[np.argsort(-a)[:K]]), all=metrics(xyz), alpha_at_K=float(np.sort(a)[::-1][min(K, len(a)) - 1]), **image(r / 'evaluation_common475/summary.json'))
rf = ROOT / 'ranerf' / 'eval_ranerf_goes.json'
if rf.exists(): out['RaNeRF'] = json.loads(rf.read_text())
for d in range(1, 17):
    r = ROOT / f'D{d:02d}_sh3_mask'
    if (r / 'evaluation_exit.json').exists(): out[r.name] = gs(r)
for n in ['A_fullimage_sh3', 'A_novis_sh3']:
    r = ROOT / n
    if (r / 'evaluation_exit.json').exists() and (r / 'training/checkpoints/final.pt').exists(): out[n] = gs(r)
(ROOT / 'collected.json').write_text(json.dumps(dict(K=K, results=out), indent=1))
rows = ['method,N,cd_l1,cd_l2,hd95,acc,comp,cov01,psnr,ssim']
for name in ['P_sh3_mask', 'B_Pei_uniform', 'B_ISARGS', 'B_RNSplat_base', 'RaNeRF', 'A_fullimage_sh3', 'A_novis_sh3']:
    if name not in out: continue
    m = out[name].get('q050') or out[name].get('topK') or out[name].get('top%d' % K) or {}
    if not m: continue
    rows.append(','.join([name, str(m['N'])] + ['%.6f' % m[k] for k in ('cd_l1', 'cd_l2', 'hd95', 'acc', 'comp', 'cov01')] + ['%.4f' % out[name].get(k, float('nan')) for k in ('psnr', 'ssim')]))
(ROOT / 'table1_goes.csv').write_text('\n'.join(rows) + '\n')
rows = ['run,N50,cdl1_50,cdl2_50,hd95_50,cov_50,all_cdl1,psnr']
for d in range(1, 17):
    n = f'D{d:02d}_sh3_mask'
    if n in out:
        a = out[n]['q050']; rows.append(','.join([n, str(a['N'])] + ['%.6f' % a[k] for k in ('cd_l1', 'cd_l2', 'hd95', 'cov01')] + ['%.6f' % out[n]['all']['cd_l1'], '%.3f' % out[n].get('psnr', float('nan'))]))
(ROOT / 'daily_goes.csv').write_text('\n'.join(rows) + '\n')
print(open(ROOT / 'table1_goes.csv').read()); print(open(ROOT / 'daily_goes.csv').read())
