"""Pei et al. Eqs.5-9, BSAT budget adaptation (not official reproduction)."""
import argparse
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from radar_gs.data import load_dataset
from radar_gs.geometry import projection_coordinates, projection_jacobian
from radar_gs.losses import ssim
from radar_gs.io import export_ply


def sh3(d):
    x,y,z=d.unbind(-1); xx,yy,zz=x*x,y*y,z*z
    return torch.stack((torch.ones_like(x)*.28209479177387814,
      -.4886025119029199*y,.4886025119029199*z,-.4886025119029199*x,
      1.0925484305920792*x*y,-1.0925484305920792*y*z,
      .31539156525252005*(2*zz-xx-yy),-1.0925484305920792*x*z,
      .5462742152960396*(xx-yy),-.5900435899266435*y*(3*xx-yy),
      2.890611442640554*x*y*z,-.4570457994644658*y*(4*zz-xx-yy),
      .3731763325901154*z*(2*zz-3*xx-3*yy),
      -.4570457994644658*x*(4*zz-xx-yy),1.445305721320277*z*(xx-yy),
      -.5900435899266435*x*(xx-3*yy)),dim=-1)


def rotation(q):
    w,x,y,z=torch.nn.functional.normalize(q,dim=-1).unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
      2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
      2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),dim=-1).reshape(-1,3,3)


class Scene(nn.Module):
    def __init__(self, means, bbox=6.):
        super().__init__(); assert means.ndim==2 and means.shape[1]==3; n=len(means); self.bbox=bbox; self.degree=0
        self.means=nn.Parameter(means.clone())
        self.log_scales=nn.Parameter(torch.full((n,3),math.log(.04),device=means.device))
        q=torch.zeros(n,4,device=means.device);q[:,0]=1
        self.quaternions=nn.Parameter(q)
        self.support_logits=nn.Parameter(torch.full((n,),math.log(.1/.9),device=means.device))
        self.sh_dc=nn.Parameter(torch.zeros(n,1,device=means.device))
        self.sh_rest=nn.Parameter(torch.zeros(n,15,device=means.device))
    @property
    def count(self):return len(self.means)
    def scales(self):return self.log_scales.exp().clamp_min(1e-4)
    def normals(self):return rotation(self.quaternions)[:,:,2]
    def covariance(self):
        r=rotation(self.quaternions);return r@torch.diag_embed(self.scales().square())@r.transpose(-1,-2)
    def support_probability(self,temperature=1.):return self.support_logits.sigmoid()
    def amplitude(self,kr):
        k=(self.degree+1)**2
        return (.5+(torch.cat((self.sh_dc,self.sh_rest),-1)[:,:k]*sh3(kr/kr.norm())[:k]).sum(-1)).clamp_min(0.)


class LinearRenderer(nn.Module):
    """Unnormalized Gaussian footprints + linear accumulation, no visibility.

    Three-sigma truncation follows the standard GS footprint convention.
    Dynamic rectangles retain the footprint as scales change. No fixed9x9 crop.
    """
    def __init__(self,height,width,*args,**kwargs):
        super().__init__();self.height=height;self.width=width;self.use_cuda=True
    def forward(self,scene,kr,kd,dr,dd,**kwargs):
        kr=kr.to(scene.means);kd=kd.to(scene.means)
        row,col,_=projection_coordinates(scene.means,kr,kd,self.height,self.width,dr,dd)
        p=projection_jacobian(kr,kd,dr,dd)
        centers=torch.stack((row,col),dim=-1)
        if centers.requires_grad:centers.retain_grad()
        row,col=centers.unbind(-1)
        cov=p@scene.covariance()@p.T
        cov=cov+torch.eye(2,device=row.device)[None]*.3 # standard GS low-pass variance
        inv=torch.linalg.inv(cov)
        radii=torch.ceil(3*(.5*(cov.detach()[:,0,0]+cov.detach()[:,1,1]+((cov.detach()[:,0,0]-cov.detach()[:,1,1]).square()+4*cov.detach()[:,0,1].square()).sqrt())).sqrt()).long().clamp_min(1)
        radii=2**torch.ceil(torch.log2(radii.float())).long()
        # ponytail: PyTorch bucket splatting; CUDA rasterizer when measured cost warrants it.
        assert torch.isfinite(cov).all(), 'Nonfinite covariance'
        inside=(row+radii>=0)&(row-radii<self.height)&(col+radii>=0)&(col-radii<self.width)
        amp=scene.amplitude(kr);alpha=scene.support_probability()
        if self.use_cuda:
            from pei_linear_cuda_wide import rasterize
            conics=torch.stack((inv[:,0,0],inv[:,0,1],inv[:,1,1]),-1)
            image=rasterize(centers,conics,alpha*amp,radii,self.height,self.width)
            return SimpleNamespace(image=image,raw_image=image,support_map=image.clamp(0,1),overlap_map=image*0,projected_centers=centers)
        image=row.new_zeros(self.height*self.width)
        for radius in torch.unique(radii[inside]).tolist():
            ids=torch.where(inside&(radii==radius))[0]
            axis=torch.arange(-radius,radius+1,device=row.device)
            rr,cc=torch.meshgrid(axis,axis,indexing='ij');rr=rr.flatten();cc=cc.flatten()
            chunk=max(1,min(512,500000//len(rr)))
            for start in range(0,len(ids),chunk):
                ix=ids[start:start+chunk]
                pr=row[ix,None].floor().long()+rr;pc=col[ix,None].floor().long()+cc
                delta=torch.stack((pr-row[ix,None],pc-col[ix,None]),-1)
                maha=torch.einsum('gki,gij,gkj->gk',delta,inv[ix],delta)
                valid=(pr>=0)&(pr<self.height)&(pc>=0)&(pc<self.width)&(maha<=9)
                values=torch.exp(-.5*maha)*valid*(alpha[ix]*amp[ix])[:,None]
                pix=pr.clamp(0,self.height-1)*self.width+pc.clamp(0,self.width-1)
                image=image.scatter_add(0,pix.flatten(),values.flatten())
        image=image.reshape(self.height,self.width)
        return SimpleNamespace(image=image,raw_image=image,support_map=image.clamp(0,1),overlap_map=image*0,projected_centers=centers)


NAMES=['means','log_scales','quaternions','support_logits','sh_dc','sh_rest']
LRS=[1.6e-4,.005,.001,.05,.0025,.000125]


def make_optimizer(scene):
    return torch.optim.Adam([{'params':[getattr(scene,n)],'lr':lr,'name':n} for n,lr in zip(NAMES,LRS)],eps=1e-15)


@torch.no_grad()
def change_topology(scene,opt,keep,parents,new_means,new_scales):
    for group in opt.param_groups:
        name=group['name'];old=getattr(scene,name)
        data=torch.cat((old[keep],old[parents]),0)
        if name=='means':data[len(keep):]=new_means
        if name=='log_scales':data[len(keep):]=new_scales
        new=nn.Parameter(data);state=opt.state.pop(old,{})
        for key in ['exp_avg','exp_avg_sq']:
            if key in state:state[key]=torch.cat((state[key][keep],torch.zeros_like(old[parents])),0)
        group['params']=[new];opt.state[new]=state;setattr(scene,name,new)


@torch.no_grad()
def refine(scene,opt,grad,it,cap):
    alpha=scene.support_probability();scale=scene.scales().amax(-1)
    keep=torch.where(alpha>=.004)[0]
    if not len(keep):raise RuntimeError('All Gaussians pruned; scientific failure, not silently rescued')
    candidates=keep[grad[keep]>=.0002]
    budget=max(0,cap-len(keep));candidates=candidates[torch.argsort(grad[candidates],descending=True)]
    chosen=[];cost=0
    for ix in candidates.tolist():
        increment=1
        if cost+increment>budget:break
        chosen.append(ix);cost+=increment
    chosen=torch.tensor(chosen,device=keep.device,dtype=torch.long)
    split=chosen[scale[chosen]>.01*(2*scene.bbox)]
    clone=chosen[scale[chosen]<=.01*(2*scene.bbox)]
    keep=keep[~torch.isin(keep,split)]
    parents=torch.cat((clone,split.repeat_interleave(2)))
    means=scene.means[parents].clone();ls=scene.log_scales[parents].clone()
    if len(split):
        ix=parents[len(clone):]
        jitter=torch.randn_like(means[len(clone):])*scene.scales()[ix]
        means[len(clone):]+=(rotation(scene.quaternions[ix])@jitter[:,:,None]).squeeze(-1)
        ls[len(clone):]-=math.log(1.6)
    change_topology(scene,opt,keep,parents,means,ls)
    print('REFINE',it,'count',scene.count,'clone',len(clone),'split',len(split),flush=True)


def load_scene(path,device):
    ck=torch.load(path,map_location=device,weights_only=False)
    scene=Scene(ck['model']['means'],ck['arguments']['bbox']).to(device)
    scene.load_state_dict(ck['model']);scene.degree=ck['degree'];return scene,ck


def train(args):
    root=Path(args.out);root.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(Path(args.config).read_text());cfg.update(iters=args.iters,render_gamma=1.,disable_visibility=True,final_support_temperature=1.)
    cfg['implementation']='Pei equation-based BSAT budget adaptation; NOT official/native paper result'
    cfg['initialization']=args.init;cfg['budget_deviations']=['10k initial/cap20k vs69740/unrestricted','BSAT coordinates and frozen preprocessing vsJason3','12k/batch8 vs30k/unspecified','densify500-6000 for12k','standard GS3sigma, .3pixel variance','unspecified hyperparameters use3DGS defaults','no native CUDA timing equivalence']
    (root/'run_config.json').write_text(json.dumps(cfg,indent=2))
    torch.manual_seed(42);np.random.seed(42);random.seed(42)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2)
    views,report=load_dataset(cfg['data_root'],image_size=256,native_dr=cfg['native_dr'],native_dd=cfg['native_dd'],encounter_names=cfg['encounter_names'],frames_per_encounter=1,images_per_encounter=1,background_threshold=cfg['background_threshold'],normalization_percentile=cfg['normalization_percentile'])
    assert len(views)==1903
    (root/'selected_views.json').write_text(json.dumps(report,indent=2))
    if args.init=='train_points':
        z=np.load(cfg['init_path']);print('INIT_KEYS',z.files,flush=True);means=torch.tensor(z['points'][:,:3] if 'points' in z else z['means'],device='cuda',dtype=torch.float32)
        assert len(means)==10000 and len(torch.unique(means,dim=0))==10000
    else:means=(torch.rand(10000,3,device='cuda')*2-1)*cfg['bbox']
    scene=Scene(means,cfg['bbox']).cuda();renderer=LinearRenderer(256,256).cuda();opt=make_optimizer(scene)
    export_ply(root/'gaussians_initial.ply',scene)
    gen=random.Random(42);history=[];grad_sum=torch.zeros(scene.count,device='cuda');grad_count=0
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t0=time.perf_counter()
    for it in range(1,args.iters+1):
        scene.degree=min(3,it//1000);opt.zero_grad(set_to_none=True);batch=gen.sample(range(len(views)),8);lossval=0.
        for idx in batch:
            v=views[idx];rendered=renderer(scene,v.kr,v.kd,v.dr,v.dd);pred=rendered.image
            target=v.image.cuda();loss=.8*(pred-target).abs().mean()+.2*(1-ssim(pred,target))
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss iteration{it}')
            (loss/8).backward();lossval+=float(loss.detach())/8
            grad_sum+=rendered.projected_centers.grad.detach().norm(dim=-1)*128*8;grad_count+=1
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in scene.parameters()):raise RuntimeError('Nonfinite gradient')
        
        # Original GS position LR decay. Others remain constant.
        opt.param_groups[0]['lr']=1.6e-4*(.01**(it/args.iters))
        opt.step()
        if 500<=it<=args.iters//2 and it%100==0:
            refine(scene,opt,grad_sum/max(grad_count,1),it,20000)
            grad_sum=torch.zeros(scene.count,device='cuda');grad_count=0
        if it%3000==0 and it<=args.iters//2:
            with torch.no_grad():scene.support_logits.copy_(torch.logit(scene.support_probability().clamp_max(.01).clamp_min(1e-6)))
            state=opt.state[scene.support_logits]
            for k in ['exp_avg','exp_avg_sq']:
                if k in state:state[k].zero_()
        if it==1 or it%100==0:
            row={'iteration':it,'loss':lossval,'count':scene.count,'degree':scene.degree,'seconds':time.perf_counter()-t0};history.append(row);print('PEI_TRAIN',json.dumps(row),flush=True)
        if it%1000==0 or it==args.iters:
            (root/'checkpoints').mkdir(exist_ok=True)
            ck={'iteration':it,'arguments':cfg,'degree':scene.degree,'model':scene.state_dict(),'optimizer':opt.state_dict(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'sampling_rng':gen.getstate(),'grad_sum':grad_sum,'grad_count':grad_count}
            torch.save(ck,root/'checkpoints'/('final.pt' if it==args.iters else f'iter_{it:06d}.pt'))
    torch.cuda.synchronize();timing={'iterations':args.iters,'selected_views':1903,'views_per_batch':8,'final_gaussians':scene.count,'total_seconds':time.perf_counter()-t0,'peak_gpu_allocated_gb':torch.cuda.max_memory_allocated()/2**30,'peak_gpu_reserved_gb':torch.cuda.max_memory_reserved()/2**30}
    export_ply(root/'gaussians_final.ply',scene)
    (root/'training_history.json').write_text(json.dumps(history,indent=2));(root/'timing_report.json').write_text(json.dumps(timing,indent=2))
    print('PEI_COMPLETE',json.dumps(timing),flush=True)


def check():
    torch.manual_seed(0)
    s=Scene(torch.tensor([[0.,0.,0.]],device='cuda'));r=LinearRenderer(15,15)
    kr=torch.tensor([0.,0.,1.],device='cuda');kd=torch.tensor([1.,0.,0.],device='cuda')
    v=r(s,kr,kd,.1,.1).image
    assert torch.allclose(v[7,7],torch.tensor(.05,device='cuda'),atol=1e-6)
    covariance=s.covariance();assert torch.allclose(covariance,torch.eye(3,device='cuda')[None]*.04**2,atol=1e-7)
    two=Scene(s.means.detach().repeat(2,1));assert torch.allclose(r(two,kr,kd,.1,.1).image,2*v,atol=1e-6)
    v.sum().backward();assert s.support_logits.grad.abs().sum()>0 and torch.isfinite(s.log_scales.grad).all()
    # Positional finite difference at an off-center point.
    s=Scene(torch.tensor([[.017,.011,.023]],device='cuda'));v=r(s,kr,kd,.1,.1).image
    objective=(v*torch.arange(225,device='cuda').reshape(15,15)).sum();g=torch.autograd.grad(objective,s.means)[0][0,0]
    vals=[]
    for delta in [-1e-4,1e-4]:
        with torch.no_grad():s.means[0,0]=.017+delta
        vals.append((r(s,kr,kd,.1,.1).image*torch.arange(225,device='cuda').reshape(15,15)).sum().detach())
    fd=(vals[1]-vals[0])/2e-4;assert torch.allclose(g,fd,rtol=.025,atol=.05),(g,fd)
    r.use_cuda=False;reference=r(s,kr,kd,.1,.1).image
    reference_grad=torch.autograd.grad(reference.square().sum(),list(s.parameters()),allow_unused=True)
    r.use_cuda=True;fast=r(s,kr,kd,.1,.1).image
    fast_grad=torch.autograd.grad(fast.square().sum(),list(s.parameters()),allow_unused=True)
    assert torch.allclose(reference,fast,atol=1e-6,rtol=1e-5)
    for x,y in zip(reference_grad,fast_grad):
        if x is not None:assert torch.allclose(x,y,atol=1e-5,rtol=1e-4),(x,y)
    print('PEI_CHECK_PASS: covariance, superposition, finite difference, CUDA/reference forward+all parameter gradients',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--config');p.add_argument('--out');p.add_argument('--init',choices=['uniform','train_points']);p.add_argument('--iters',type=int,default=12000);a=p.parse_args()
    check() if a.check else train(a)

