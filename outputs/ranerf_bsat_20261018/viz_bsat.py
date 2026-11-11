import sys
from pathlib import Path
import numpy as np, torch, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
I = Path('/rds/homes/g/gey/Yao/programs/ISAR-3DGS'); R = I / 'outputs/ranerf_bsat_20261018'; RUN = R / 'ranerf_seed42'
GT = np.load(I / 'outputs/bsat_observation_days_20261004/skynet_gt_blender_5000.npy')
# 1. training renders
steps = ['0005000', '0015000', '0030000', '0045000']
fig, ax = plt.subplots(1, 5, figsize=(12, 2.8))
ax[0].imshow(np.asarray(Image.open(RUN / 'render_preview/step_0045000_gt.png').convert('L')), cmap='gray'); ax[0].set_title('ground truth', fontsize=9)
for A, s in zip(ax[1:], steps):
    A.imshow(np.asarray(Image.open(RUN / ('render_preview/step_%s_pred.png' % s)).convert('L')), cmap='gray'); A.set_title('RaNeRF, iter %d' % int(s), fontsize=9)
for A in ax: A.set_axis_off()
fig.tight_layout(); fig.savefig(R / 'bsat_ranerf_renders.png', dpi=110); plt.close(fig)
# 2. geometry
g = np.load(RUN / 'mesh_fixed_20261019/density_grid.npy'); res = g.shape[0]
def xyz(idx): return -6 + idx / (res - 1) * 12
order = np.argsort(g.ravel())[::-1]
top = lambda K: xyz(np.array(np.unravel_index(order[:K], g.shape)).T)
p92 = xyz(np.argwhere(g >= np.percentile(g, 92.0)))
m = torch.load(I / 'outputs/q_support_20261016/ent300_seed42/training/checkpoints/final.pt', map_location='cpu', weights_only=False)['model']
prop = m['means'].numpy()[torch.sigmoid(m['support_logits'] / 0.35).numpy() >= 0.5]
rows = [('Proposed (q>=0.5)', prop), ('RaNeRF, densest 11,472 voxels', top(11472)), ('RaNeRF, densest 20,000 voxels', top(20000)), ('RaNeRF, default (>=92nd pct)', p92)]
lo = GT.min(0) - 0.6; hi = GT.max(0) + 0.6
fig, ax = plt.subplots(4, 3, figsize=(10, 9))
for i, (lab, p) in enumerate(rows):
    for j, (a, b) in enumerate([(0, 1), (0, 2), (1, 2)]):
        A = ax[i, j]; A.scatter(GT[:, a], GT[:, b], s=0.3, c='0.6', lw=0)
        sub = p if len(p) < 60000 else p[np.random.default_rng(0).choice(len(p), 60000, replace=False)]
        A.scatter(sub[:, a], sub[:, b], s=0.6, c='#2a78d6', lw=0, alpha=.6)
        A.set_xlim(lo[a], hi[a]); A.set_ylim(lo[b], hi[b]); A.set_aspect('equal'); A.tick_params(labelsize=6)
        A.set_xlabel('xyz'[a] + ' (m)', fontsize=7); A.set_ylabel('xyz'[b] + ' (m)', fontsize=7)
        out = int((~((p >= lo) & (p <= hi)).all(1)).sum())
        if j == 1: A.set_title('%s: N=%d, %d outside' % (lab, len(p), out), fontsize=8)
fig.tight_layout(); fig.savefig(R / 'bsat_ranerf_pointclouds.png', dpi=110); plt.close(fig)
# 3. default set over the full box
fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
sub = p92[np.random.default_rng(0).choice(len(p92), 80000, replace=False)]
for A, (a, b) in zip(ax, [(0, 1), (0, 2), (1, 2)]):
    A.scatter(sub[:, a], sub[:, b], s=0.5, c='#2a78d6', lw=0, alpha=.3); A.scatter(GT[:, a], GT[:, b], s=0.2, c='k', lw=0)
    A.set_xlim(-6, 6); A.set_ylim(-6, 6); A.set_aspect('equal'); A.set_xlabel('xyz'[a]); A.set_ylabel('xyz'[b])
fig.suptitle('RaNeRF default voxel set (>=92nd density percentile, 566k voxels; blue) vs GT (black), full +/-6 m box', fontsize=9)
fig.tight_layout(); fig.savefig(R / 'bsat_ranerf_default_fullbox.png', dpi=100)
print('done', len(prop))
