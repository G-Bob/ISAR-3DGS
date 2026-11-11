import math
from pathlib import Path
import numpy as np, torch
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from hybrid_worker import install
install('sh3')
from radar_gs.io import load_scene
from radar_gs.data import load_dataset
from radar_gs.presets import load_encounter_batch_preset
from radar_gs.losses import compress_amplitude
import evaluate_ablation as ev
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if dev.type == 'cuda': torch.cuda.set_per_process_memory_fraction(0.2)
Q = Path('outputs/q_main_20261017'); old = Path('outputs/bsat_observation_days_20261004')
DAYS = [1, 3, 5, 7, 9, 12, 16]
CD = {1: 0.148, 3: 0.102, 5: 0.088, 7: 0.066, 9: 0.052, 12: 0.052, 16: 0.049}
GT = np.load(old / 'skynet_gt_blender_5000.npy'); tree = cKDTree(GT)
preset = load_encounter_batch_preset(old / 'evaluation_preset.json', allow_incomplete=True)
views = None; R = ev.RadarVisibilityRenderer
renders, clouds = [], []
for d in DAYS:
    tr = Q / f'D{d:02d}_sh3_mask' / 'training'
    scene, ck = load_scene(tr / 'checkpoints/final.pt', dev); cfg = ck['arguments']
    if views is None:
        views, _ = load_dataset(cfg['data_root'], image_size=256, native_dr=cfg['native_dr'], native_dd=cfg['native_dd'], encounter_names=preset['encounter_names'], frames_per_encounter=1, background_threshold=cfg['background_threshold'], normalization_percentile=cfg['normalization_percentile'])
        v = views[len(views) // 2]; tgt = v.image.cpu().numpy()
    rend = R(256, 256, cfg['depth_bins'], cfg['kernel_radius'], cfg['minimum_pixel_variance'], cfg['opacity_gain'], cfg.get('visibility_pixel_size', 0.0), cfg.get('support_composition', 'noisy_or')).to(dev)
    T = cfg.get('final_support_temperature', 0.35); gamma = float(cfg.get('render_gamma', 1.0))
    with torch.no_grad():
        q = scene.support_probability(T)
        out = rend(scene, v.kr, v.kd, v.dr, v.dd, support_probability=q, support_temperature=T)
        renders.append(compress_amplitude(out.image, gamma).cpu().numpy())
        P = scene.means[q >= 0.5].cpu().numpy()
    clouds.append(P); print('day', d, 'N', len(P), flush=True)
ys, xs = np.where(tgt > 0.05); cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
h = min(128, int(max(ys.max() - ys.min(), xs.max() - xs.min()) * 0.65) + 8)
sl = (slice(max(cy - h, 0), cy + h), slice(max(cx - h, 0), cx + h))
n = len(DAYS) + 1
fig = plt.figure(figsize=(1.55 * n, 3.55))
def lab(ax, t): ax.set_title(t, fontsize=9, pad=3)
ax = fig.add_subplot(2, n, 1); ax.imshow(tgt[sl], cmap='magma', vmin=0, vmax=1); ax.axis('off'); lab(ax, 'Ground truth')
ax = fig.add_subplot(2, n, n + 1, projection='3d')
ax.scatter(GT[:, 0], GT[:, 1], GT[:, 2], s=0.15, c='0.45', depthshade=False); lab(ax, 'CAD reference')
lim = np.abs(GT).max() * 0.5
def style(ax):
    ax.view_init(elev=22, azim=-58); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
style(ax)
for k, d in enumerate(DAYS):
    ax = fig.add_subplot(2, n, k + 2); ax.imshow(renders[k][sl], cmap='magma', vmin=0, vmax=1); ax.axis('off'); lab(ax, f'Day {d}')
    P = clouds[k]; dist = tree.query(P)[0]
    ax = fig.add_subplot(2, n, n + k + 2, projection='3d')
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=0.15, c=dist, cmap='viridis', vmin=0, vmax=0.3, depthshade=False); style(ax)
    lab(ax, f'CD-L1 {CD[d]:.3f} m')
fig.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.01, wspace=0.02, hspace=0.06)
fig.savefig(Q / 'figs_labelled/acq_days_visual.png', dpi=250)
print('saved')
