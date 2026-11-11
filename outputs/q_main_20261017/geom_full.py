import json, glob, datetime as dt
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
DS = Path('/rds/homes/g/gey/Yao/programs/GEIST-6/datasets/bsat')
cpis = sorted(p.parent for p in DS.glob('cpi_*/_COMPLETE.json'))
P = lambda s: dt.datetime.fromisoformat(s.replace('Z', '+00:00'))
start, end, los, rng, arcl = [], [], [], [], []
for c in cpis:
    k = json.load(open(c / '_COMPLETE.json')); m = json.load(open(c / 'static_los_metadata.json'))['preset_encounter']
    start.append(P(k['cpi_start_utc'])); end.append(P(k['cpi_end_utc']))
    los.append(m['range_axis_center_rtn']); rng.append(m['range_km']); arcl.append(np.degrees(m['unit_sphere_arc_length_rad']))
o = np.argsort(start); cpis = [cpis[i] for i in o]
start = [start[i] for i in o]; end = [end[i] for i in o]
los, rng, arcl = np.array(los)[o], np.array(rng)[o], np.array(arcl)[o]
t0 = start[0]
day = np.array([(s - t0).total_seconds() / 86400 for s in start])
dur = np.array([(e - s).total_seconds() for s, e in zip(start, end)])
step = np.degrees(np.arccos(np.clip((los[1:] * los[:-1]).sum(1), -1, 1)))
arc = np.concatenate([[0], np.cumsum(np.minimum(step, 1.929))]) + 1.929
gaps = np.diff(day) * 24
np.savez('acq_geometry_all.npz', day=day, dur=dur, rng=rng, los=los, cpi=np.array([c.name for c in cpis]))
print('N', len(cpis), 'span_days', round((end[-1] - t0).total_seconds() / 86400, 3), 'first', cpis[0].name, start[0].isoformat(), 'last', cpis[-1].name, end[-1].isoformat())
print('dur_min', round(dur.min() / 60, 1), round(np.median(dur) / 60, 1), round(dur.max() / 60, 1), 'rate_deg_s', round((arcl / dur).min(), 5), round((arcl / dur).max(), 4))
print('range_km', round(rng.min(), 1), round(rng.max(), 1), 'arc_total', round(arc[-1], 0), 'largest_gaps_h', np.round(np.sort(gaps)[-4:], 2), 'at_day', np.round(day[np.argsort(gaps)[-4:]], 2))
plt.rcParams.update({'font.size': 7})
fig = plt.figure(figsize=(7.16, 2.35))
ax = fig.add_subplot(1, 3, 1, projection='3d')
sc = ax.scatter(los[:, 0], los[:, 1], los[:, 2], c=day, cmap='viridis', s=1.0, depthshade=False)
ax.set_xlabel('R', labelpad=-8); ax.set_ylabel('T', labelpad=-8); ax.set_zlabel('N', labelpad=-8); ax.set_box_aspect((1, 1, 1))
ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.set_zticks([-1, 0, 1])
for a in (ax.xaxis, ax.yaxis, ax.zaxis): a.set_tick_params(labelsize=5, pad=-3)
cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.12); cb.set_label('Elapsed day', fontsize=6); cb.ax.tick_params(labelsize=5)
ax.set_title('(a)', fontsize=7, loc='left')
ax2 = fig.add_subplot(1, 3, 2)
ax2.scatter(day, rng, s=0.8, c='#2a78d6', label='range')
ax2.set_xlabel('Elapsed day'); ax2.set_ylabel('Range (km)'); ax2.grid(alpha=0.25, lw=0.5)
ax3 = ax2.twinx(); ax3.plot(day, arc / 1000, color='0.3', lw=0.9, ls='--', label='accumulated LOS arc')
ax3.set_ylabel(r'Accumulated LOS arc ($10^3$ deg)')
h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax3.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, fontsize=5.5, frameon=False, loc='upper center', markerscale=4)
ax2.set_title('(b)', fontsize=7, loc='left')
ax4 = fig.add_subplot(1, 3, 3)
ax4.scatter(day, dur / 60, s=0.8, c='#e8710a')
ax4.set_yscale('log'); ax4.set_xlabel('Elapsed day'); ax4.set_ylabel(r'Orbital time per $1.929^\circ$ CPI (min)')
ax4.grid(alpha=0.25, lw=0.5, which='both')
sec = ax4.secondary_yaxis('right', functions=(lambda T: 1.929 / np.maximum(T, 1e-6), lambda w: 1.929 / np.maximum(w, 1e-6)))
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter, FormatStrFormatter
ax4.yaxis.set_major_locator(FixedLocator([4, 6, 10, 20, 40])); ax4.yaxis.set_major_formatter(FormatStrFormatter('%g')); ax4.yaxis.set_minor_locator(NullLocator())
sec.yaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.2, 0.4])); sec.yaxis.set_major_formatter(FormatStrFormatter('%g')); sec.yaxis.set_minor_locator(NullLocator())
sec.set_ylabel(r'LOS rate (deg/min)')
ax4.set_title('(c)', fontsize=7, loc='left')
fig.tight_layout(w_pad=1.2)
fig.savefig('figs_labelled/bsat_geometry_window_full.png', dpi=300)
print('saved')
