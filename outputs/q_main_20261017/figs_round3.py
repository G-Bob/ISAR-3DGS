import json, csv, datetime as dt
from pathlib import Path
import numpy as np
src = open('outputs/q_main_20261017/figs.py').read().split("if __name__")[0]
ns = {}; exec(src, ns)
plt, VIEWS, panel, centres, Q, P = ns['plt'], ns['VIEWS'], ns['panel'], ns['centres'], ns['Q'], ns['P']
from plyread import read_ply
out = Path('outputs/q_main_20261017/figs_labelled'); L = 'abcdefghi'
def lab(ax, k): ax.set_title('(%s)' % L[k], fontsize=8, loc='left')
# Fig. 7: proposed q>=0.5 versus Pei uniform with the same number of points (top-K by its own weight)
v = read_ply(P / 'P30_Pei_uniform_wide_restart' / 'gaussians_final.ply')
xyz = np.stack([v['x'], v['y'], v['z']], 1).astype(float); w = v['target_support_probability'].astype(float)
pei = xyz[np.argsort(-w)[:11472]]; prop = centres(Q / 'ent300_seed42', 0.5)
fig, axs = plt.subplots(2, 2, figsize=(7.16, 3.7), squeeze=False); k = 0
for i, p in enumerate([prop, pei]):
    for j, (a, b, xl, yl) in enumerate(VIEWS[:2]):
        panel(axs[i, j], p, a, b, xl, yl, ''); lab(axs[i, j], k); k += 1
fig.tight_layout(h_pad=0.6, w_pad=0.6); pass  # fig.savefig(out / 'orthogonal_comparison_matchedK.png', dpi=250); plt.close(fig)
print('pei outside', ns['outside'](pei), 'prop outside', ns['outside'](prop))
# Fig. 2: geometry of the rendered 16.078-day window only
D = Path('/rds/homes/g/gey/Yao/programs/GEIST-6/datasets/bsat')
rows = list(csv.DictReader(open('outputs/bsat_observation_days_20261004/cpi_membership.csv')))
t0 = dt.datetime.fromisoformat(rows[0]['start_utc'])
day, los, rng, tr = [], [], [], []
for r in rows:
    m = json.load(open(D / r['cpi'] / 'static_los_metadata.json'))['preset_encounter']
    day.append((dt.datetime.fromisoformat(r['start_utc']) - t0).total_seconds() / 86400)
    los.append(m['range_axis_center_rtn']); rng.append(m['range_km']); tr.append(r['split'] == 'train')
day, los, rng, tr = map(np.array, (day, los, rng, tr))
step = np.degrees(np.arccos(np.clip((los[1:] * los[:-1]).sum(1), -1, 1)))
arc = np.concatenate([[0], np.cumsum(np.minimum(step, 1.929))]) + 1.929
print('n', len(day), 'span', day.max(), 'range min/max', rng.min(), rng.max(), 'day of min', day[rng.argmin()], 'arc', arc[-1], 'raw arc', step.sum())
plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize=(7.16, 3.0))
ax = fig.add_subplot(1, 2, 1, projection='3d')
sc = ax.scatter(los[:, 0], los[:, 1], los[:, 2], c=day, cmap='viridis', s=1.5, depthshade=False)
ax.set_xlabel('R'); ax.set_ylabel('T'); ax.set_zlabel('N'); ax.set_box_aspect((1, 1, 1))
for a in (ax.xaxis, ax.yaxis, ax.zaxis): a.set_tick_params(labelsize=6, pad=0)
cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.2); cb.set_label('Elapsed day', fontsize=7); cb.ax.tick_params(labelsize=6)
ax.set_title('(a)', fontsize=8, loc='left')
ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(day, rng, s=1.5, c='#2a78d6', label='CPI range')
ax2.set_xlabel('Elapsed day'); ax2.set_ylabel('Range (km)'); ax2.grid(alpha=0.25, lw=0.5)
ax3 = ax2.twinx(); ax3.plot(day, arc / 1000, color='0.3', lw=1.0, ls='--', label='accumulated LOS arc')
ax3.set_ylabel(r'Accumulated LOS arc ($10^3$ deg)')
h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax3.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, fontsize=6, frameon=False, loc='upper center', markerscale=4)
ax2.set_title('(b)', fontsize=8, loc='left')
fig.tight_layout(); fig.savefig(out / 'bsat_geometry_window.png', dpi=250)
print('done')
