"""Paper figure: RaNeRF on BSAT. (a) training image, (b) RaNeRF render, (c) default voxel set over the full box, (d)-(f) densest 11,472 voxels, (g)-(i) densest 20,000 voxels."""
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
I = Path('/rds/homes/g/gey/Yao/programs/ISAR-3DGS'); R = I / 'outputs/ranerf_bsat_20261018'; RUN = R / 'ranerf_seed42'
GT = np.load(I / 'outputs/bsat_observation_days_20261004/skynet_gt_blender_5000.npy')
g = np.load(RUN / 'mesh_fixed_20261019/density_grid.npy'); res = g.shape[0]
xyz = lambda idx: -6 + idx / (res - 1) * 12
order = np.argsort(g.ravel())[::-1]
top = lambda K: xyz(np.array(np.unravel_index(order[:K], g.shape)).T)
p92 = xyz(np.argwhere(g >= np.percentile(g, 92.0)))
lo = GT.min(0) - 0.6; hi = GT.max(0) + 0.6
L = iter('abcdefghi')
plt.rcParams.update({'font.size': 7})
fig = plt.figure(figsize=(7.16, 6.6))
gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 1, 1])
def lab(A): A.set_title('(%s)' % next(L), fontsize=8, loc='left')
for j, name in enumerate(['step_0045000_gt.png', 'step_0045000_pred.png']):
    A = fig.add_subplot(gs[0, j]); A.imshow(np.asarray(Image.open(RUN / 'render_preview' / name).convert('L')), cmap='gray'); A.set_axis_off(); lab(A)
A = fig.add_subplot(gs[0, 2]); sub = p92[np.random.default_rng(0).choice(len(p92), 80000, replace=False)]
A.scatter(sub[:, 0], sub[:, 2], s=0.3, c='#2a78d6', lw=0, alpha=.3); A.scatter(GT[:, 0], GT[:, 2], s=0.2, c='k', lw=0)
A.set_xlim(-6, 6); A.set_ylim(-6, 6); A.set_aspect('equal'); A.set_xlabel('x (m)'); A.set_ylabel('z (m)'); lab(A)
for i, K in enumerate([11472, 20000], start=1):
    p = top(K)
    for j, (a, b) in enumerate([(0, 1), (0, 2), (1, 2)]):
        A = fig.add_subplot(gs[i, j]); A.scatter(GT[:, a], GT[:, b], s=0.3, c='0.6', lw=0); A.scatter(p[:, a], p[:, b], s=0.5, c='#2a78d6', lw=0, alpha=.7)
        A.set_xlim(lo[a], hi[a]); A.set_ylim(lo[b], hi[b]); A.set_aspect('equal'); A.tick_params(labelsize=6)
        A.set_xlabel('xyz'[a] + ' (m)'); A.set_ylabel('xyz'[b] + ' (m)'); lab(A)
fig.tight_layout(h_pad=0.8, w_pad=0.6); fig.savefig(R / 'fig_ranerf_bsat.png', dpi=250)
print('done')
