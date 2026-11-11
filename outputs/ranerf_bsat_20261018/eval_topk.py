import json, numpy as np, sys
sys.argv=['x']
src=open('/rds/homes/g/gey/Yao/programs/ISAR-3DGS/outputs/ranerf_bsat_20261018/eval_ranerf.py').read().split('mesh_dir = next')[0]
ns={}; exec(src, ns)
R=ns['R']; RUN=ns['RUN']; metrics=ns['metrics']; BBOX=6.0
g=np.load(RUN/'mesh'/'density_grid.npy'); res=g.shape[0]; f=g.ravel()
pct={p: float(np.percentile(f,p)) for p in (50,90,92,99,99.5,99.9,99.95,99.99)}
out=dict(max=float(f.max()), mean=float(f.mean()), pct=pct)
order=np.argsort(f)[::-1]
for K in (5000,11500,20000,50000,100000):
    idx=np.array(np.unravel_index(order[:K], g.shape)).T
    pts=-BBOX+idx.astype(np.float32)/(res-1)*(2*BBOX)
    out['top%d'%K]=metrics(pts)
(R/'eval_ranerf_topk.json').write_text(json.dumps(out,indent=1))
print('max',out['max'],'pct',{k:'%.2e'%v for k,v in pct.items()})
for K in (5000,11500,20000,50000,100000):
    m=out['top%d'%K]; print(K,'L1 %.4f L2 %.4f HD %.3f acc %.4f comp %.4f cov %.1f'%(m['cd_l1'],m['cd_l2'],m['hd95'],m['acc'],m['comp'],100*m['cov01']))
