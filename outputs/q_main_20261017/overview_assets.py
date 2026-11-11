import json, csv, datetime as dt
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree
OUT = Path('outputs/q_main_20261017/figs_labelled/overview'); OUT.mkdir(parents=True, exist_ok=True)
DS = Path('/rds/homes/g/gey/Yao/programs/GEIST-6/datasets/bsat')
GT = np.load('outputs/bsat_observation_days_20261004/skynet_gt_blender_5000.npy'); TREE = cKDTree(GT)
rows = list(csv.DictReader(open('outputs/bsat_observation_days_20261004/cpi_membership.csv')))
train = [r['cpi'] for r in rows if r['split'] == 'train']
def isar(cpi):
    f = next((DS / cpi / 'isar_images_noaxis').glob('*.png'))
    return np.asarray(Image.open(f).convert('L'), float) / 255.0
def save_img(a, name, cmap='magma'):
    plt.imsave(OUT / name, a, cmap=cmap, vmin=0, vmax=1)
# 1. ISAR images at three elapsed times
picks = [train[int(len(train) * f)] for f in (0.05, 0.45, 0.85)]
for k, c in enumerate(picks):
    a = isar(c); save_img(np.sqrt(a / max(a.max(), 1e-6)), 'isar_%d.png' % k)
# 2. echo mask overlay for the first image (paper rule, 2-px dilation at 256 px)
a = isar(picks[0]); a = a / max(a.max(), 1e-6)
med = np.median(a); mad = np.median(np.abs(a - med)); tau = max(0.02 * a.max(), med + 4 * mad)
r = int(round(2 * a.shape[0] / 256)); m = maximum_filter(a > tau, size=2 * r + 1)
rgb = plt.get_cmap('gray')(np.sqrt(a))[..., :3] * 0.85
ov = rgb.copy(); ov[m] = 0.45 * rgb[m] + 0.55 * np.array([0.0, 0.85, 0.75])
plt.imsave(OUT / 'mask_overlay.png', ov)
print('mask frac', m.mean(), 'tau', tau)
# 3. LOS trajectory on the unit sphere
los, day = [], []
t0 = dt.datetime.fromisoformat(rows[0]['start_utc'])
for r_ in rows:
    mm = json.load(open(DS / r_['cpi'] / 'static_los_metadata.json'))['preset_encounter']
    los.append(mm['range_axis_center_rtn']); day.append((dt.datetime.fromisoformat(r_['start_utc']) - t0).total_seconds() / 86400)
los, day = np.array(los), np.array(day)
fig = plt.figure(figsize=(3, 3)); ax = fig.add_subplot(projection='3d')
u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color='0.8', lw=0.3)
ax.scatter(los[:, 0], los[:, 1], los[:, 2], c=day, cmap='viridis', s=1.2, depthshade=False)
ax.scatter([0], [0], [0], s=40, c='#e8710a', marker='s')
ax.set_box_aspect((1, 1, 1)); ax.view_init(18, -52); ax.set_axis_off()
fig.savefig(OUT / 'los_sphere.png', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0); plt.close(fig)
# 4-6. model views
m = torch.load('outputs/q_support_20261016/ent300_seed42/training/checkpoints/final.pt', map_location='cpu', weights_only=False)['model']
print(sorted(m.keys()))
mu = m['means'].numpy(); q = torch.sigmoid(m['support_logits'] / 0.35).numpy().ravel()
def view3d(p, c, name, s=0.4, cmap=None, vmin=None, vmax=None, elev=20, azim=-60):
    fig = plt.figure(figsize=(4, 3)); ax = fig.add_subplot(projection='3d')
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=c, s=s, lw=0, cmap=cmap, vmin=vmin, vmax=vmax, depthshade=False, rasterized=True)
    ax.set_xlim(-4.6, 4.6); ax.set_ylim(-2.3, 2.3); ax.set_zlim(-1.6, 1.6); ax.set_box_aspect((9.2, 4.6, 3.2))
    ax.view_init(elev, azim); ax.set_axis_off()
    fig.savefig(OUT / name, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0); plt.close(fig)
inb = (np.abs(mu[:, 0]) < 4.6) & (np.abs(mu[:, 1]) < 2.3) & (np.abs(mu[:, 2]) < 1.6)
col = np.where(q >= 0.5, '#2a78d6', '#f0a050')
view3d(mu[inb], col[inb], 'gauss_q.png', s=0.5)
ps = mu[q >= 0.5]; d = TREE.query(ps)[0]
view3d(ps, d, 'struct3d.png', s=0.5, cmap='viridis', vmin=0, vmax=0.3)
view3d(GT, '0.35', 'gt3d.png', s=0.6)
# ellipsoid close-up near a panel edge
nk = [k for k in m if 'normal' in k][0]; sk = [k for k in m if 'scale' in k][0]
n = m[nk].numpy(); n = n / np.linalg.norm(n, axis=1, keepdims=True); sc = np.exp(m[sk].numpy())
centre = np.array([1.9, 0.0, 0.45]); sel = np.where(q >= 0.5)[0]
sel = sel[np.argsort(np.linalg.norm(mu[sel] - centre, axis=1))[:90]]
fig = plt.figure(figsize=(3, 3)); ax = fig.add_subplot(projection='3d')
uu, vv = np.mgrid[0:2 * np.pi:16j, 0:np.pi:9j]
sph = np.stack([np.cos(uu) * np.sin(vv), np.sin(uu) * np.sin(vv), np.cos(vv)], -1)
cm = plt.get_cmap('cool')
for i in sel:
    ref = np.array([0, 1.0, 0]) if abs(n[i, 2]) > 0.9 else np.array([0, 0, 1.0])
    t1 = np.cross(ref, n[i]); t1 /= np.linalg.norm(t1); t2 = np.cross(n[i], t1)
    B = np.stack([t1, t2, n[i]], 1); e = (sph * (1.6 * sc[i])) @ B.T + mu[i]
    ax.plot_surface(e[..., 0], e[..., 1], e[..., 2], color=cm((mu[i, 0] - centre[0] + 0.4) / 0.8), alpha=0.85, lw=0, shade=True)
ax.set_box_aspect((1, 1, 1)); lim = 0.45
ax.set_xlim(centre[0] - lim, centre[0] + lim); ax.set_ylim(-lim, lim); ax.set_zlim(centre[2] - lim, centre[2] + lim)
ax.view_init(25, -60); ax.set_axis_off()
fig.savefig(OUT / 'ellipsoids.png', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0); plt.close(fig)
print('done', sorted(p.name for p in OUT.iterdir()))
