"""Redraw seed44 grid with short titles: method label on the middle panel only."""
import sys
from pathlib import Path
src = open('outputs/q_main_20261017/figs.py').read().split("if __name__")[0]
ns = {}; exec(src, ns)
plt, VIEWS, panel, centres, outside = ns['plt'], ns['VIEWS'], ns['panel'], ns['centres'], ns['outside']
M, Q = ns['M'], ns['Q']
rows = [('Full-image MLP', M / 'S2_mlp_none_seed44'), ('Echo-mask MLP', M / 'S4_mlp_mask_seed44'), ('Proposed', Q / 'ent300_seed44')]
fig, axs = plt.subplots(3, 3, figsize=(7.16, 6.0))
for i, (lab, run) in enumerate(rows):
    p = centres(run, 0.5)
    for j, (a, b, xl, yl) in enumerate(VIEWS):
        panel(axs[i, j], p, a, b, xl, yl, '%s (N=%d, %d outside)' % (lab, len(p), outside(p)) if j == 1 else '')
        axs[i, j].title.set_fontsize(8)
fig.tight_layout(h_pad=0.6, w_pad=0.6)
out = Path('outputs/q_main_20261017/figs/seed44_grid_short.png'); fig.savefig(out, dpi=250); print('saved', out)
