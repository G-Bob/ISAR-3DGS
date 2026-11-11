"""Point-cloud figures with empty titles and (a), (b), ... panel labels."""
from pathlib import Path
src = open('outputs/q_main_20261017/figs.py').read().split("if __name__")[0]
ns = {}; exec(src, ns)
plt, VIEWS, panel, centres, TREE, lo, hi = ns['plt'], ns['VIEWS'], ns['panel'], ns['centres'], ns['TREE'], ns['lo'], ns['hi']
M, Q, P = ns['M'], ns['Q'], ns['P']
L = 'abcdefghi'
out = Path('outputs/q_main_20261017/figs_labelled'); out.mkdir(exist_ok=True)
def lab(ax, k): ax.set_title('(%s)' % L[k], fontsize=8, loc='left')
# extraction: (a,b) all centres, (c,d) q>=0.5
fig, axs = plt.subplots(2, 2, figsize=(7.16, 3.6)); k = 0
for row, tau in enumerate([None, 0.5]):
    p = centres(Q / 'ent300_seed42', tau); d = TREE.query(p)[0]
    for col, (a, b, xl, yl) in enumerate(VIEWS[:2]):
        ax = axs[row, col]; sc = panel(ax, p, a, b, xl, yl, '', colour=d)
        ax.set_xlim(lo[a], hi[a]); ax.set_ylim(lo[b], hi[b]); ax.set_aspect('equal'); ax.set_xlabel(xl, fontsize=7); ax.set_ylabel(yl, fontsize=7); ax.tick_params(labelsize=6)
        lab(ax, k); k += 1
cb = fig.colorbar(sc, ax=axs, shrink=0.8); cb.set_label('distance to GT (m)', fontsize=7); cb.ax.tick_params(labelsize=6)
fig.savefig(out / 'extraction_proposed_seed42.png', dpi=250, bbox_inches='tight'); plt.close(fig)
def grid(rows, cols, name, h):
    fig, axs = plt.subplots(len(rows), len(cols), figsize=(7.16, h), squeeze=False); k = 0
    for i, (run, tau) in enumerate(rows):
        p = centres(run, tau)
        for j, (a, b, xl, yl) in enumerate(cols):
            panel(axs[i, j], p, a, b, xl, yl, ''); lab(axs[i, j], k); k += 1
    fig.tight_layout(h_pad=0.6, w_pad=0.6); fig.savefig(out / name, dpi=250); plt.close(fig)
grid([(Q / 'ent300_seed42', 0.5), (M / 'F8_mlp_mask_track_guard', 0.5), (P / 'P30_Pei_uniform_wide_restart', None)], VIEWS[:2], 'orthogonal_comparison_pei_uniform.png', 5.4)

print('done', sorted(p.name for p in out.iterdir()))
