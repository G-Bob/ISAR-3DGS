import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
R=Path('/rds/homes/g/gey/Yao/programs/ISAR-3DGS/outputs/ranerf_bsat_20261018'); g=np.load(R/'ranerf_seed42/mesh/density_grid.npy'); res=g.shape[0]
GT=np.load('/rds/homes/g/gey/Yao/programs/ISAR-3DGS/outputs/bsat_observation_days_20261004/skynet_gt_blender_5000.npy')
o=np.argsort(g.ravel())[::-1][:50000]; idx=np.array(np.unravel_index(o,g.shape)).T; P=-6+idx/(res-1)*12
fig,ax=plt.subplots(1,3,figsize=(12,4))
for a,(i,j) in zip(ax,[(0,1),(0,2),(1,2)]):
    a.scatter(P[:,i],P[:,j],s=0.2,c='#2a78d6',alpha=.3); a.scatter(GT[:,i],GT[:,j],s=0.3,c='k'); a.set_xlim(-6,6); a.set_ylim(-6,6); a.set_aspect('equal'); a.set_title('axes %d-%d'%(i,j))
fig.tight_layout(); fig.savefig(R/'ranerf_top50k_vs_gt.png',dpi=90)
print('centroid P',P.mean(0).round(2),'GT',GT.mean(0).round(2),'std P',P.std(0).round(2),'GT',GT.std(0).round(2))
