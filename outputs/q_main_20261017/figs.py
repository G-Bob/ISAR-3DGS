"""Point-cloud figures for the lambda_ent=0.3 / tau=0.5 main configuration. Usage: python figs.py OUTDIR [--test]"""
import sys, json
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
M = Path('outputs/q_main_20261017'); Q = Path('outputs/q_support_20261016'); P = Path('outputs/paper_reproduction')
GT = np.load('outputs/bsat_observation_days_20261004/skynet_gt_blender_5000.npy'); TREE = cKDTree(GT)
TAU = 0.5
def centres(run, tau):
    ck = run / 'training/checkpoints/final.pt'
    if ck.exists():
        m = torch.load(ck, map_location='cpu', weights_only=False)['model']
        mu = m['means'].numpy(); q = torch.sigmoid(m['support_logits'] / 0.35).numpy()
        return mu if tau is None else mu[q >= tau]
    return read_ply_xyz(run / 'gaussians_final.ply')
def read_ply_xyz(path):
    with open(path, 'rb') as fh:
        props, n, fmt = [], 0, 'ascii'
        while True:
            line = fh.readline().decode('ascii').strip()
            if line.startswith('format'): fmt = line.split()[1]
            elif line.startswith('element vertex'): n = int(line.split()[2])
            elif line.startswith('property'): props.append(line.split()[1:])
            elif line == 'end_header': break
        names = [p[-1] for p in props]
        if fmt == 'ascii':
            a = np.loadtxt(fh, max_rows=n); return a[:, [names.index('x'), names.index('y'), names.index('z')]]
        tm = {'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8', 'uchar': 'u1', 'uint8': 'u1', 'int': 'i4', 'int32': 'i4'}
        dt = np.dtype([(p[-1], ('<' if 'little' in fmt else '>') + tm[p[0]]) for p in props])
        a = np.frombuffer(fh.read(n * dt.itemsize), dtype=dt, count=n)
        return np.stack([a['x'], a['y'], a['z']], 1).astype(float)
lo = GT.min(0) - 0.6; hi = GT.max(0) + 0.6
VIEWS = [(0, 1, 'x (m)', 'y (m)'), (0, 2, 'x (m)', 'z (m)'), (1, 2, 'y (m)', 'z (m)')]
def outside(p): return int((~((p >= lo) & (p <= hi)).all(1)).sum())
def panel(ax, p, a, b, xl, yl, title, colour=None):
    ax.scatter(GT[:, a], GT[:, b], s=0.3, c='0.75', lw=0, rasterized=True)
    if colour is None: ax.scatter(p[:, a], p[:, b], s=0.3, c='#2a78d6', lw=0, rasterized=True)
    else: return ax.scatter(p[:, a], p[:, b], s=0.3, c=colour, cmap='viridis', vmin=0, vmax=0.3, lw=0, rasterized=True)
    ax.set_xlim(lo[a], hi[a]); ax.set_ylim(lo[b], hi[b]); ax.set_aspect('equal'); ax.set_xlabel(xl, fontsize=7); ax.set_ylabel(yl, fontsize=7)
    ax.set_title(title, fontsize=7); ax.tick_params(labelsize=6)
def fig_extraction(run, out):
    fig, axs = plt.subplots(2, 2, figsize=(7.16, 3.6))
    for row, (tau, lab) in enumerate([(None, 'all centres'), (TAU, 'q >= %.1f' % TAU)]):
        p = centres(run, tau); d = TREE.query(p)[0]
        for col, (a, b, xl, yl) in enumerate(VIEWS[:2]):
            sc = panel(axs[row, col], p, a, b, xl, yl, '%s: N=%d, outside %d' % (lab, len(p), outside(p)), colour=d)
            axs[row, col].set_xlim(lo[a], hi[a]); axs[row, col].set_ylim(lo[b], hi[b]); axs[row, col].set_aspect('equal')
            axs[row, col].set_xlabel(xl, fontsize=7); axs[row, col].set_ylabel(yl, fontsize=7); axs[row, col].tick_params(labelsize=6)
            axs[row, col].set_title('%s: N=%d, outside %d' % (lab, len(p), outside(p)), fontsize=7)
    cb = fig.colorbar(sc, ax=axs, shrink=0.8); cb.set_label('distance to GT (m)', fontsize=7); cb.ax.tick_params(labelsize=6)
    fig.savefig(out, dpi=250, bbox_inches='tight'); plt.close(fig)
def fig_grid(rows, out, cols=VIEWS):
    fig, axs = plt.subplots(len(rows), len(cols), figsize=(7.16, 2.0 * len(rows)), squeeze=False)
    for i, (lab, run, tau) in enumerate(rows):
        p = centres(run, tau)
        for j, (a, b, xl, yl) in enumerate(cols):
            panel(axs[i, j], p, a, b, xl, yl, '%s, N=%d, outside %d' % (lab, len(p), outside(p)))
    fig.tight_layout(); fig.savefig(out, dpi=250); plt.close(fig)
if __name__ == '__main__':
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); test = '--test' in sys.argv
    prop42, prop44 = Q / 'ent300_seed42', Q / 'ent300_seed44'
    aux = (P / 'M19_track_mask_guard_fast_restart') if test else (M / 'F8_mlp_mask_track_guard')
    if not (aux / 'training/checkpoints/final.pt').exists(): aux = Path('outputs/bsat_methods8_20261004/M19_track_mask_background_guard')
    full44 = Path('outputs/echo_mask_robustness_20261009/mlp_full_seed44') if test else M / 'S2_mlp_none_seed44'
    echo44 = Path('outputs/echo_mask_robustness_20261009/mlp_echo_seed44') if test else M / 'S4_mlp_mask_seed44'
    done = []
    for name, fn in [('extraction_proposed_seed42.png', lambda o: fig_extraction(prop42, o)),
                     ('orthogonal_comparison.png', lambda o: fig_grid([('Proposed, q>=0.5', prop42, TAU), ('Masked MLP + aux., q>=0.5', aux, TAU), ('Pei + mask/guard, all', P / 'P35_Pei_mask_guard', None)], o, VIEWS[:2])),
                     ('seed44_grid.png', lambda o: fig_grid([('Full-image MLP s44, q>=0.5', full44, TAU), ('Echo-mask MLP s44, q>=0.5', echo44, TAU), ('Proposed s44, q>=0.5', prop44, TAU)], o))]:
        try: fn(out / name); done.append(name)
        except Exception as e: print('FAILED', name, repr(e), flush=True)
    print('figures', done, flush=True)
