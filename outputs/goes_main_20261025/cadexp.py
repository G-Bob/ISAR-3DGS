import trimesh, numpy as np
from scipy.spatial import cKDTree
R=np.array([[1,0,0],[0,0,-1],[0,1,0]],float)
C='/rds/homes/g/gey/Yao/programs/GEIST-6/CAD/'
O='outputs/goes_main_20261025/figs/'
GTS={'skynet':np.load('outputs/bsat_observation_days_20261004/skynet_gt_blender_5000.npy'),'goes':np.load('outputs/goes_main_20261025/goes_gt_blender_5000.npy')}
for tag,f in [('skynet','skynet5d/skynet5d.glb'),('skynet_half','skynet5d/skynet5d_half_scale.glb'),('goes','GOES-R/GOES-R-0.4.glb')]:
    m=trimesh.load(C+f,force='mesh')
    v=np.asarray(m.vertices)@R.T
    m2=trimesh.Trimesh(v,m.faces,process=False)
    s=m2.sample(300000)
    gt=GTS[tag.split('_')[0]]
    d=cKDTree(s).query(gt)[0]
    print(tag,len(m.faces),'gt->mesh med %.4f p95 %.4f'%(np.median(d),np.percentile(d,95)))
    if tag!='skynet_half' or True: np.savez(O+'cad_'+tag+'.npz',v=v.astype(np.float32),f=np.asarray(m.faces,np.int32))