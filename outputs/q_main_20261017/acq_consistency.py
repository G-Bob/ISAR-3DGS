import json, csv, datetime as dt
from pathlib import Path
import numpy as np
DS = Path('/rds/homes/g/gey/Yao/programs/GEIST-6/datasets/bsat')
rows = list(csv.DictReader(open('outputs/bsat_observation_days_20261004/cpi_membership.csv')))
GT = np.load('outputs/bsat_observation_days_20261004/skynet_gt_blender_5000.npy')
lam = 299792458 / 300e9; N = 751; prf_replay = 1400.0
t0 = dt.datetime.fromisoformat(rows[0]['start_utc'])
day, dur, arc, rng, omg = [], [], [], [], []
for r in rows:
    m = json.load(open(DS / r['cpi'] / 'static_los_metadata.json'))['preset_encounter']
    s, e = dt.datetime.fromisoformat(r['start_utc']), dt.datetime.fromisoformat(r['end_utc'])
    day.append((s - t0).total_seconds() / 86400); dur.append((e - s).total_seconds())
    arc.append(np.degrees(m['unit_sphere_arc_length_rad'])); rng.append(m['range_km']); omg.append(m['omega_deg_s'])
day, dur, arc, rng, omg = map(np.array, (day, dur, arc, rng, omg))
rate = arc / dur
ext = GT.max(0) - GT.min(0); D = float(np.linalg.norm(ext))
dtheta = np.radians(arc.mean()); step = dtheta / (N - 1)
out = dict(n=len(dur), lam_mm=lam * 1e3, arc_deg=[arc.min(), arc.max()], omega_replay=[omg.min(), omg.max()],
           T_replay_s=N / prf_replay, extent_m=ext.tolist(), Dmax_m=D,
           cross_res_m=lam / (2 * dtheta), unamb_extent_m=lam / (2 * step),
           doppler_bw_replay_Hz=2 * np.radians(omg.mean()) * D / lam)
for name, v in [('dur_min', dur / 60), ('rate_deg_s', rate), ('prf_needed_Hz', (N - 1) / dur), ('doppler_bw_real_Hz', 2 * np.radians(rate) * D / lam), ('range_km', rng)]:
    out[name] = dict(min=float(v.min()), p5=float(np.percentile(v, 5)), median=float(np.median(v)), p95=float(np.percentile(v, 95)), max=float(v.max()))
out['corr_rate_range'] = float(np.corrcoef(rate, 1 / rng)[0, 1])
json.dump(out, open('outputs/q_main_20261017/acq_consistency.json', 'w'), indent=1)
np.savez('outputs/q_main_20261017/acq_consistency_percpi.npz', day=day, dur=dur, rate=rate, rng=rng)
for k, v in out.items(): print(k, v)
