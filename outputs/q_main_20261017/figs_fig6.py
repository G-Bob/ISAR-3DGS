"""Point-cloud figures with empty titles and (a), (b), ... panel labels."""
from pathlib import Path
src = open('outputs/q_main_20261017/figs.py').read().split("if __name__")[0]
ns = {}; exec(src, ns)
plt, VIEWS, panel, centres, TREE, lo, hi = ns['plt'], ns['VIEWS'], ns['panel'], ns['centres'], ns['TREE'], ns['lo'], ns['hi']
M, Q, P = ns['M'], ns['Q'], ns['P']
L = 'abcdefghi'
out = Path('outputs/q_main_20261017/figs_labelled'); out.mkdir(exist_ok=True)
def lab(ax, k): ax.set_title('(%s)' % L[k], fontsize=8, loc='left')
def grid(rows, cols, name, h):
    fig, axs = plt.subplots(len(rows), len(cols), figsize=(7.16, h), squeeze=False); k = 0
    for i, (run, tau) in enumerate(rows):
        p = centres(run, tau)
        for j, (a, b, xl, yl) in enumerate(cols):
            panel(axs[i, j], p, a, b, xl, yl, ''); lab(axs[i, j], k); k += 1
    fig.tight_layout(h_pad=0.6, w_pad=0.6); fig.savefig(out / name, dpi=250); plt.close(fig)
grid([(Q / 'ent300_seed42', 0.5), (M / 'F5_mlp_mask', 0.5), (P / 'P30_Pei_uniform_wide_restart', None)], VIEWS[:2], 'orthogonal_comparison_pei_uniform.png', 5.4)

print('done', sorted(p.name for p in out.iterdir()))
