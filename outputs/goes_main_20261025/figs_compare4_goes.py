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
fig, axs = plt.subplots(4, 3, figsize=(7.16, 6.95), gridspec_kw={'width_ratios': w}, squeeze=False)
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
(G / 'figs').mkdir(exist_ok=True); fig.savefig(G / 'figs' / 'comparison_four_methods_goes.png', dpi=300)
print('done')
