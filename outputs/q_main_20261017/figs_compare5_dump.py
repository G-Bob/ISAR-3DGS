from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
src = open('outputs/q_main_20261017/figs.py').read().split("if __name__")[0]
ns = {}; exec(src, ns)
plt, VIEWS, centres, Q, P, GT, TREE, lo, hi, outside = (ns[k] for k in ['plt', 'VIEWS', 'centres', 'Q', 'P', 'GT', 'TREE', 'lo', 'hi', 'outside'])
from plyread import read_ply
K = 11472
def topk_ply(run):
    v = read_ply(P / run / 'gaussians_final.ply')
    xyz = np.stack([v['x'], v['y'], v['z']], 1).astype(float)
    return xyz[np.argsort(-v['target_support_probability'].astype(float))[:K]]
g = np.load('outputs/ranerf_bsat_20261018/ranerf_seed42/mesh_fixed_20261019/density_grid.npy'); res = g.shape[0]
idx = np.array(np.unravel_index(np.argsort(g.ravel())[::-1][:K], g.shape)).T
ranerf = -6.0 + idx.astype(float) / (res - 1) * 12.0
rows = [centres(Q / 'ent300_seed42', 0.5), topk_ply('P30_Pei_uniform_wide_restart'), topk_ply('P32_ISARGS_complex_continuum'),
        topk_ply('P33_RNS_alpha_base'), ranerf]
names = ['proposed', 'pei', 'isargs', 'rnsplat_gain', 'ranerf']
for n, p in zip(names, rows):
    dp = TREE.query(p)[0]; dg = cKDTree(p).query(GT)[0]
    print(n, len(p), 'cd1 %.5f cov %.2f outside %d' % (0.5 * (dp.mean() + dg.mean()), 100 * (dg < 0.1).mean(), outside(p)))
plt.rcParams.update({'font.size': 7})
w = [hi[0] - lo[0], hi[0] - lo[0], hi[1] - lo[1]]
np.savez('outputs/goes_main_20261025/figs/pts_skynet.npz', GT=GT, names=np.array(names), **{n:p for n,p in zip(names,rows)})
