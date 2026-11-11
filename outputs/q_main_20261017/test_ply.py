import sys
sys.argv = ['x', 'outputs/q_main_20261017/figs_test']
src = open('outputs/q_main_20261017/figs.py').read().split("if __name__")[0]
ns = {}; exec(src, ns)
p = ns['read_ply_xyz']('outputs/paper_reproduction/P35_Pei_mask_guard/gaussians_final.ply'); print('pei', p.shape, p.min(0).round(2), p.max(0).round(2))
from pathlib import Path
ns['fig_grid']([('Proposed, q>=0.5', ns['Q'] / 'ent300_seed42', 0.5), ('Masked SH3 + aux (old M30), q>=0.5', ns['P'] / 'M30_M19_SH3', 0.5), ('Pei + mask/guard, all', ns['P'] / 'P35_Pei_mask_guard', None)], Path('outputs/q_main_20261017/figs_test/orthogonal_comparison.png'), ns['VIEWS'][:2]); print('ortho ok')
