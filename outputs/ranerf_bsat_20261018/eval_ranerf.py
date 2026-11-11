"""Evaluate RaNeRF (BSAT protocol) with the same GT and metrics as the Gaussian methods. Outputs eval_ranerf.json."""
import json, itertools, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
R = Path('/rds/homes/g/gey/Yao/programs/ISAR-3DGS/outputs/ranerf_bsat_20261018'); RUN = R / 'ranerf_seed42'
GT = np.load('/rds/homes/g/gey/Yao/programs/ISAR-3DGS/outputs/bsat_observation_days_20261004/skynet_gt_blender_5000.npy'); TG = cKDTree(GT)
BBOX = 6.0
def metrics(P):
    P = np.asarray(P, float)
    if len(P) == 0: return dict(N=0)
    dpg = TG.query(P)[0]; dgp = cKDTree(P).query(GT)[0]
    return dict(N=int(len(P)), cd_l1=float(.5 * (dpg.mean() + dgp.mean())), cd_l2=float(.5 * ((dpg ** 2).mean() + (dgp ** 2).mean())),
                hd95=float(max(np.percentile(dpg, 95), np.percentile(dgp, 95))), acc=float(dpg.mean()), comp=float(dgp.mean()), cov01=float((dgp <= .1).mean()))
def read_ply_vertices(path):
    with open(path) as f:
        n = 0
        while True:
            l = f.readline().strip()
            if l.startswith('element vertex'): n = int(l.split()[2])
            if l == 'end_header': break
        return np.loadtxt(f, max_rows=n)[:, :3]
def otsu(x, bins=256):
    h, e = np.histogram(x, bins=bins); c = (e[:-1] + e[1:]) / 2; w = np.cumsum(h); m = np.cumsum(h * c); W = w[-1]; Mt = m[-1]
    with np.errstate(all='ignore'): s = (Mt * w / W - m) ** 2 / (w * (W - w))
    return float(c[np.nanargmax(s)])
mesh_dir = next(p for p in [RUN / 'mesh', RUN] if (p / 'density_grid.npy').exists())
grid = np.load(mesh_dir / 'density_grid.npy'); res = grid.shape[0]
def vox(level):
    idx = np.argwhere(grid >= level); return -BBOX + idx.astype(np.float32) / (res - 1) * (2 * BBOX)
out = dict(run=str(RUN), grid_shape=list(grid.shape))
lv92 = float(np.percentile(grid, 92.0)); out['voxels_p92'] = dict(level=lv92, **metrics(vox(lv92)))
lg = np.log(np.maximum(grid, 1e-8)); lo = otsu(lg.ravel()); out['voxels_otsu_log'] = dict(level=float(np.exp(lo)), **metrics(vox(np.exp(lo))))
if (mesh_dir / 'final_mesh.ply').exists():
    V = read_ply_vertices(mesh_dir / 'final_mesh.ply'); out['mesh_vertices'] = metrics(V)
    flips = {}
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product([1, -1], repeat=3):
            flips['%s%s' % (perm, sg)] = metrics(V[:, perm] * np.array(sg))['cd_l1']
    best = sorted(flips.items(), key=lambda kv: kv[1])[:4]; out['frame_check_mesh_cd_l1'] = dict(identity=flips['(0, 1, 2)(1, 1, 1)'], best=best)
(R / 'eval_ranerf.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out)[:1500])
