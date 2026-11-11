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
fig, axs = plt.subplots(5, 3, figsize=(7.16, 8.6), gridspec_kw={'width_ratios': w}, squeeze=False)
L = 'abcdefghijklmno'; k = 0
for i, p in enumerate(rows):
    d = TREE.query(p)[0]
    for j, (a, b, xl, yl) in enumerate(VIEWS):
        ax = axs[i, j]
        ax.scatter(GT[:, a], GT[:, b], s=0.3, c='0.8', lw=0, rasterized=True)
        sc = ax.scatter(p[:, a], p[:, b], s=0.3, c=d, cmap='viridis', vmin=0, vmax=0.3, lw=0, rasterized=True)
        ax.set_xlim(lo[a], hi[a]); ax.set_ylim(lo[b], hi[b]); ax.set_aspect('equal')
        ax.set_xlabel(xl, fontsize=6); ax.set_ylabel(yl, fontsize=6); ax.tick_params(labelsize=5)
        ax.set_title('(%s)' % L[k], fontsize=7, loc='left'); k += 1
fig.tight_layout(rect=(0, 0.07, 1, 1), h_pad=0.4, w_pad=0.4)
cax = fig.add_axes([0.3, 0.045, 0.4, 0.011])
cb = fig.colorbar(sc, cax=cax, orientation='horizontal'); cb.set_label('Distance to the nearest GT sample (m)', fontsize=7); cb.ax.tick_params(labelsize=6)
fig.savefig('outputs/q_main_20261017/figs_labelled/comparison_five_methods.png', dpi=300)
print('done')
