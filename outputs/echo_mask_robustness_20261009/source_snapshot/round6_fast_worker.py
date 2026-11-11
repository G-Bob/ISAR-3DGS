import json, os, runpy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import radar_gs.initialization as init
import radar_gs.trainer as trainer
from radar_gs.losses import charbonnier

ROOT=Path('outputs/bsat_methods8_20261004')
MASK=os.environ.get('R6_MASK')=='1'
TRACK=os.environ.get('R6_TRACK','1')=='1'
GUARD=os.environ.get('R6_GUARD')=='1'
DATA=json.loads((ROOT/'M15_multicpi_tracks/tracks.json').read_text())

def reliable_mask(target):
    # Echo support proxy, not an occlusion or physical visibility certificate.
    med=target.median(); mad=(target-med).abs().median()
    threshold=torch.maximum(target.max()*.02,med+4*mad)
    seed=(target>threshold).float()
    return F.max_pool2d(seed[None,None],5,1,2)[0,0].detach()

def masked_loss(prediction,target,lambda_ssim,lambda_log=.2,charbonnier_epsilon=.001,log_gain=20.,weight=None):
    p=prediction.clamp(0,1);t=target.clamp(0,1);m=reliable_mask(t)
    if weight is not None:m=m*weight.to(m).clamp(0,1)
    robust=charbonnier(p-t,charbonnier_epsilon,m)
    log=charbonnier(torch.log1p(log_gain*p)-torch.log1p(log_gain*t),charbonnier_epsilon,m)
    # Only SSIM windows wholly inside accepted pixels receive gradients.
    avg=lambda x:F.avg_pool2d(x[None,None],7,1,3)[0,0]
    a,b=avg(p),avg(t);vx,vy=avg(p*p)-a*a,avg(t*t)-b*b;cov=avg(p*t)-a*b
    score=((2*a*b+.01**2)*(2*cov+.03**2))/((a*a+b*b+.01**2)*(vx+vy+.03**2)).clamp_min(1e-8)
    valid=(avg(m)>=1-1e-6).float()
    masked_loss.calls=getattr(masked_loss,'calls',0)+1
    if masked_loss.calls%800==1:print('R6_MASK','calls',masked_loss.calls,'pixel_fraction',float(m.mean()),'ssim_window_fraction',float(valid.mean()),flush=True)
    ssim=(score*valid).sum()/valid.sum().clamp_min(1)
    ssim_loss=((1-score)*valid).sum()/valid.sum().clamp_min(1)
    bg=1-m
    guard=.02*((p-t).abs()*bg).sum()/bg.sum().clamp_min(1) if GUARD else p.new_zeros(())
    return robust+lambda_log*log+lambda_ssim*ssim_loss+guard,dict(charbonnier=robust,log_charbonnier=log,ssim=ssim)

class TrackConstraint:
    def __init__(self):self.cache=None
    def __call__(self,scene,iteration):
        if not TRACK:return scene.means.new_zeros(())
        device=scene.means.device
        if self.cache is None:
            anchors=[];matrices=[];pixels=[];owners=[];weights=[]
            for k,t in enumerate(DATA['tracks']):
                anchors.append(t['point'])
                for v,j in t['observations']:
                    g=DATA['geometry'][v]
                    matrices.append(np.stack([-np.array(g['kr'])/g['dr'],np.array(g['kd'])/g['dd']]))
                    # Images use common shape; infer from actual training view below.
                    pixels.append(DATA['detections'][v][j]);owners.append(k)
                    weights.append(1/(1+t['rms_px']**2)/len(t['observations']))
            self.cache=[torch.tensor(np.array(x),device=device,dtype=torch.float32) for x in [anchors,matrices,pixels,weights]]+[torch.tensor(owners,device=device)]
        anchors,A,uv,w,owner=self.cache
        # Reassociate geometrically after splitting/pruning; no persistent identity claim.
        with torch.no_grad():
            dist=torch.cdist(anchors,scene.means.detach());used=set();ids=[];valid=[]
            candidates=dist.argsort(dim=1)[:,:len(anchors)]
            orders=candidates.cpu().tolist(); distances=dist.gather(1,candidates).cpu().tolist()
            for order,ds in zip(orders,distances):
                rank=next(k for k,j in enumerate(order) if j not in used);idx=order[rank];used.add(idx);ids.append(idx);valid.append(ds[rank]<=.15)
            ids=torch.tensor(ids,device=device);valid=torch.tensor(valid,device=device)
        center=torch.tensor([(self.shape[0]-1)/2,(self.shape[1]-1)/2],device=device)
        residual=torch.einsum('nij,nj->ni',A,scene.means[ids[owner]])+center-uv
        weight=w*valid[owner];value=F.smooth_l1_loss(residual,torch.zeros_like(residual),reduction='none',beta=1).mean(1)
        loss=(value*weight).sum()/weight.sum().clamp_min(1e-8)
        ramp=min(1.,max(0.,(iteration-500)/500))
        if iteration%100==0:print('R6_TRACK',iteration,'active',int(valid.sum()),'loss',float(loss.detach()),'ramp',ramp,flush=True)
        return .01*ramp*loss

def exact(path,count,bbox,seed):
    a=np.load(path)['points'];assert len(a)==count
    return torch.tensor(a[:,:3]),torch.tensor(a[:,3:6])

if __name__=='__main__':
    torch.cuda.set_per_process_memory_fraction(.20)
    import inspect
    src=Path(trainer.__file__).read_text();needle='        total = image_loss\n';assert src.count(needle)==1
    # Inspect the actual target dimensions without changing view sampling.
    src=src.replace(needle,'        R6.shape = target.shape\n        total = image_loss + R6(scene, iteration)\n')
    trainer.R6=TrackConstraint()
    if MASK:
        trainer.reconstruction_loss=masked_loss
        p=torch.ones(32,32,requires_grad=True);t=torch.zeros(32,32);t[10:22,10:22]=1
        loss,_=masked_loss(p,t,.2);loss.backward();assert torch.isfinite(loss) and (GUARD or p.grad[:4].abs().max()==0)
    exec(compile(src,trainer.__file__,'exec'),trainer.__dict__)
    # exec reinstates the import; install mask after executing module definitions.
    if MASK:trainer.reconstruction_loss=masked_loss
    init.pointcloud_surface=exact
    print('R6_READY masked',MASK,'tracking',TRACK,'guard',GUARD,'memory_fraction',.20,'tracks',len(DATA['tracks']),flush=True)
    runpy.run_path('run_reconstruction.py',run_name='__main__')



