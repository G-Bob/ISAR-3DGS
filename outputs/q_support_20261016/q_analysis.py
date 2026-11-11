"""q diagnostics on finished SH3 checkpoints: threshold sweep, q-vs-distance, AUROC vs amplitude scores, matched-K, GT-free Otsu threshold."""
import json,sys
from pathlib import Path
import numpy as np, torch
from scipy.spatial import cKDTree
sys.path.insert(0,'outputs/echo_mask_robustness_20261009/source_snapshot')
from pei_bsat_wide import sh3
DATA=Path('/rds/homes/g/gey/Yao/programs/GEIST-6/datasets/bsat')
GT=np.load('outputs/bsat_observation_days_20261004/skynet_gt_blender_5000.npy')
TRAIN=Path('outputs/echo_mask_robustness_20261009/sh3_echo_seed43/train_cpis.txt').read_text().split()
def view_dirs():
    k=[]
    for n in TRAIN:
        p=DATA/n/'view_angle.npy'
        if not p.exists(): p=DATA/n/'Es'/'view_angle.npy'
        k.append(np.load(p)[0,:3])
    k=np.asarray(k,dtype=np.float32); return torch.from_numpy(k/np.linalg.norm(k,axis=1,keepdims=True))
def metrics(P,tree_g,gt):
    if len(P)==0: return dict(N=0)
    dpg=tree_g.query(P)[0]; dgp=cKDTree(P).query(gt)[0]
    return dict(N=int(len(P)),cd_l1=float(.5*(dpg.mean()+dgp.mean())),cd_l2=float(.5*((dpg**2).mean()+(dgp**2).mean())),
                hd95=float(max(np.percentile(dpg,95),np.percentile(dgp,95))),acc=float(dpg.mean()),comp=float(dgp.mean()),cov01=float((dgp<=.1).mean()))
def auroc(score,label):
    order=np.argsort(score); r=np.empty(len(score)); r[order]=np.arange(1,len(score)+1)
    n1=label.sum(); n0=len(label)-n1; return float((r[label].sum()-n1*(n1+1)/2)/(n1*n0))
def otsu(x,bins=256):
    h,e=np.histogram(x,bins=bins,range=(0,1)); c=(e[:-1]+e[1:])/2; w=np.cumsum(h); m=np.cumsum(h*c); W=w[-1]; M=m[-1]
    with np.errstate(all='ignore'): s=(M*w/W-m)**2/(w*(W-w))
    return float(c[np.nanargmax(s)])
def analyse(ckpt,gt,K,mlp=False):
    m=torch.load(ckpt,map_location='cpu',weights_only=False)['model']
    mu=m['means'].numpy(); q=torch.sigmoid(m['support_logits']/0.35).numpy(); f=m['features']
    if not mlp:
        a=torch.sigmoid(f@sh3(K).T)  # [G,V]
        amp_mean=a.mean(1).numpy(); amp_max=a.max(1).values.numpy()
    else: amp_mean=amp_max=q
    tg=cKDTree(gt); d=tg.query(mu)[0]
    out=dict(ckpt=str(ckpt),q_hist=np.histogram(q,bins=10,range=(0,1))[0].tolist(),q_mid_fraction=float(((q>.1)&(q<.9)).mean()))
    out['sweep']={f'{t:.2f}':metrics(mu[q>=t],tg,gt) for t in np.arange(.05,.96,.05)}
    out['all']=metrics(mu,tg,gt)
    bins=np.linspace(0,1,11); idx=np.clip(np.digitize(q,bins)-1,0,9)
    out['q_bins']=[dict(lo=float(bins[i]),n=int((idx==i).sum()),median_d=float(np.median(d[idx==i])) if (idx==i).any() else None,frac_lt005=float((d[idx==i]<.05).mean()) if (idx==i).any() else None) for i in range(10)]
    scores=dict(q=q) if mlp else dict(q=q,amp_mean=amp_mean,amp_max=amp_max,q_x_amp=q*amp_mean)
    out['spearman_q_amp']=float(np.corrcoef(np.argsort(np.argsort(q)),np.argsort(np.argsort(amp_mean)))[0,1])
    for thr in (.05,.1):
        lab=d<thr; out[f'auroc_d{thr}']={k:auroc(v,lab) for k,v in scores.items()}
    k=int((q>=.3).sum()); out['matchedK']={n:metrics(mu[np.argsort(-v)[:k]],tg,gt) for n,v in scores.items()}
    t=otsu(q); out['otsu']=dict(threshold=t,**metrics(mu[q>=t],tg,gt))
    return out
if __name__=='__main__':
    K=view_dirs(); res={}
    for spec in sys.argv[2:]:
        lab,ck=spec.split('=',1); res[lab]=analyse(ck,GT,K,lab.startswith('mlp'))
        r=res[lab]; print(lab,'q030',r['sweep']['0.30']['cd_l1'],'mid',round(r['q_mid_fraction'],3),'auc005',{k:round(v,3) for k,v in r['auroc_d0.05'].items()},'otsu',round(r['otsu']['threshold'],3),round(r['otsu']['cd_l1'],5),flush=True)
    Path(sys.argv[1]).write_text(json.dumps(res,indent=1))
