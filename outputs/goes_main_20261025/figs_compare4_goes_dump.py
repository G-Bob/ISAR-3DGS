from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
src = open('outputs/q_main_20261017/figs.py').read().split("if __name__")[0].replace("np.load('outputs/bsat_observation_days_20261004/skynet_gt_blender_5000.npy')", "np.load('outputs/goes_main_20261025/goes_gt_blender_5000.npy')")
assert 'goes_gt_blender_5000' in src
import json
G = Path('outputs/goes_main_20261025')
ns = {}; exec(src, ns)
plt, VIEWS, centres, Q, P, GT, TREE, lo, hi, outside = (ns[k] for k in ['plt', 'VIEWS', 'centres', 'Q', 'P', 'GT', 'TREE', 'lo', 'hi', 'outside'])
from plyread import read_ply
K = json.load(open(G / 'collected.json'))['K']
def topk_ply(run):
    v = read_ply(G / run / 'gaussians_final.ply')
    xyz = np.stack([v['x'], v['y'], v['z']], 1).astype(float)
    return xyz[np.argsort(-v['target_support_probability'].astype(float))[:K]]
rows = [centres(G / 'P_sh3_mask', 0.5), topk_ply('B_Pei_uniform/run'), topk_ply('B_ISARGS'),
        topk_ply('B_RNSplat_base')]
names = ['proposed', 'pei', 'isargs', 'rnsplat_gain']
for n, p in zip(names, rows):
    dp = TREE.query(p)[0]; dg = cKDTree(p).query(GT)[0]
    print(n, len(p), 'cd1 %.5f cov %.2f outside %d' % (0.5 * (dp.mean() + dg.mean()), 100 * (dg < 0.1).mean(), outside(p)))
plt.rcParams.update({'font.size': 7})
w = [hi[0] - lo[0], hi[0] - lo[0], hi[1] - lo[1]]
np.savez('outputs/goes_main_20261025/figs/pts_goes.npz', GT=GT, names=np.array(names), **{n:p for n,p in zip(names,rows)})
