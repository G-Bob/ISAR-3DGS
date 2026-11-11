"""RaNeRF evaluation with the export's index order corrected: extract_density_grid uses meshgrid(indexing='ij') with (yy, xx, zz), so grid[i, j, k] holds the density at (x=xs[j], y=xs[i], z=xs[k])."""
import json, numpy as np, sys
sys.argv=['x']
src=open('/rds/homes/g/gey/Yao/programs/ISAR-3DGS/outputs/ranerf_bsat_20261018/eval_ranerf.py').read().split('mesh_dir = next')[0]
ns={}; exec(src, ns)
R, RUN, metrics, otsu, read_ply = ns['R'], ns['RUN'], ns['metrics'], ns['otsu'], ns['read_ply_vertices']
BBOX=6.0
g=np.load(RUN/'mesh'/'density_grid.npy').transpose(1, 0, 2)   # -> (x, y, z)
res=g.shape[0]
def to_xyz(idx): return -BBOX + idx.astype(np.float32)/(res-1)*(2*BBOX)
out=dict(note='grid transposed (1,0,2) to undo the ij-indexing of extract_density_grid', checkpoint_step=45000)
lv=float(np.percentile(g,92.0)); out['voxels_p92']=dict(level=lv, **metrics(to_xyz(np.argwhere(g>=lv))))
lo=float(np.exp(otsu(np.log(np.maximum(g,1e-8)).ravel()))); out['voxels_otsu_log']=dict(level=lo, **metrics(to_xyz(np.argwhere(g>=lo))))
V=read_ply(RUN/'mesh'/'final_mesh.ply'); out['mesh_vertices_default_level']=metrics(V[:, [1, 0, 2]])
order=np.argsort(g.ravel())[::-1]
for K in (11472, 20000, 50000, 100000):
    out['top%d'%K]=metrics(to_xyz(np.array(np.unravel_index(order[:K], g.shape)).T))
try:
    from skimage import measure
    for p in (99.0, 99.5, 99.9):
        lvp=float(np.percentile(g,p)); verts,_,_,_=measure.marching_cubes(g, level=lvp); out['mesh_p%s'%p]=dict(level=lvp, **metrics(to_xyz(verts)))
except Exception as e: out['mesh_err']=repr(e)
(R/'eval_ranerf_fixed.json').write_text(json.dumps(out,indent=1))
for k,v in out.items():
    if isinstance(v,dict) and 'cd_l1' in v: print(k, v['N'], 'L1 %.4f L2 %.5f HD %.3f acc %.4f comp %.4f cov %.1f'%(v['cd_l1'],v['cd_l2'],v['hd95'],v['acc'],v['comp'],100*v['cov01']))
    elif k=='mesh_err': print(v)
