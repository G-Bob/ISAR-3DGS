"""Independent alpha reference, physical sanity gates and actual training CPI."""
import json, math, os, time
from pathlib import Path
import torch
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2)
from alpha_cuda import rasterize


def alpha_reference(m,q,a,c,h,w):
    yy,xx=torch.meshgrid(torch.arange(h,device=m.device),torch.arange(w,device=m.device),indexing='ij')
    t=torch.ones(h,w,device=m.device);out=torch.zeros(h,w,3,device=m.device)
    for j in range(len(m)):
        d=yy-m[j,0];e=xx-m[j,1];z=q[j,0]*d*d+2*q[j,1]*d*e+q[j,2]*e*e
        aa=(a[j]*torch.exp(-.5*z)*(z<=9)).clamp_max(.99)
        out=out+t[:,:,None]*aa[:,:,None]*c[j];t=t*(1-aa)
    return out


def alpha_check():
    torch.manual_seed(4321);dev='cuda'
    m=(torch.rand(67,2,device=dev)*5+2).requires_grad_();q=torch.tensor([[.8,.1,.6]],device=dev).repeat(67,1).requires_grad_();a=(torch.rand(67,device=dev)*.6+.1).requires_grad_();c=torch.rand(67,3,device=dev,requires_grad=True)
    y=rasterize(m,q,a,c,11,13);ref=alpha_reference(m,q,a,c,11,13);w=torch.randn_like(y)
    g=torch.autograd.grad((y*w).sum(),[m,q,a,c]);gr=torch.autograd.grad((ref*w).sum(),[m,q,a,c])
    err=float((y-ref).abs().max());errors=[float((x-z).abs().max()) for x,z in zip(g,gr)]
    assert err<2e-5 and max(errors)<3e-4,(err,errors)
    return dict(forward_max_error=err,gradient_max_errors=errors)


def check():
    mode=os.environ['PAPER_MODE'];report={'mode':mode,'alpha_reference':alpha_check()}
    import adapted_papers as adapted
    import pei_bsat_wide as base
    from radar_gs.data import load_dataset
    adapted.install()
    cfg=json.loads(Path('outputs/paper_reproduction/P31_Pei_trainpoints_wide_restart/run_config.json').read_text())
    views,_=load_dataset(cfg['data_root'],image_size=256,native_dr=cfg['native_dr'],native_dd=cfg['native_dd'],encounter_names=cfg['encounter_names'][:1],frames_per_encounter=1,images_per_encounter=1,background_threshold=cfg['background_threshold'],normalization_percentile=cfg['normalization_percentile'])
    import numpy as np
    coords=torch.tensor(np.load(cfg['init_path'])['points'][:,:3],device='cuda')
    scene=base.Scene(coords,cfg['bbox']).cuda();scene.degree=3;renderer=base.LinearRenderer(256,256).cuda();v=views[0];target=v.image.cuda();opt=base.make_optimizer(scene)
    def loss(rendered):
        if mode=='pei_mask':return base.mask_guard_loss(rendered.image,target,.1,.2)[0]
        if mode.startswith('rns'):return .8*(rendered.rgb-adapted.rgb_target(target)).abs().mean()+.2*sum(1-base.ssim(rendered.rgb[:,:,j],adapted.rgb_target(target)[:,:,j]) for j in range(3))/3
        return .8*(rendered.image-target).abs().mean()+.2*(1-base.ssim(rendered.image,target))
    for _ in range(2):
        opt.zero_grad(set_to_none=True);l=loss(renderer(scene,v.kr,v.kd,v.dr,v.dd));l.backward();opt.step()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    opt.zero_grad(set_to_none=True);l=loss(renderer(scene,v.kr,v.kd,v.dr,v.dd));l.backward();torch.cuda.synchronize()
    assert torch.isfinite(l)
    grads={n:float(p.grad.norm()) if p.grad is not None else None for n,p in scene.named_parameters()}
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in scene.parameters())
    if mode=='isargs':assert grads['imag_dc']>0 and grads['imag_rest']>0
    if mode=='rns_rsu':assert grads['rsu_band']>0 and grads['rsu_beta']>0
    report.update(count=scene.count,loss=float(l),gradient_norms=grads,view_forward_backward_seconds=time.perf_counter()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    # Exercise optimizer-preserving cloning, including extra complex/RGB fields.
    keep=torch.arange(scene.count,device='cuda');parents=keep[:2];base.change_topology(scene,opt,keep,parents,scene.means[:2].detach(),scene.log_scales[:2].detach());assert scene.count==10002
    opt.zero_grad(set_to_none=True);l=loss(renderer(scene,v.kr,v.kd,v.dr,v.dd));l.backward();opt.step();assert torch.isfinite(l)
    if mode=='isargs':
        trans=adapted.attenuation(scene,v.kr.cuda(),v.kd.cuda());assert (trans>0).all() and (trans<=1).all()
        report['attenuation_range']=[float(trans.min()),float(trans.max())]
    # Actual20k and broad covariance stress, not a claim of all-CPI validation.
    ids=torch.arange(scene.count,device='cuda');parents=ids[:20000-scene.count]
    base.change_topology(scene,opt,ids,parents,scene.means[parents].detach(),scene.log_scales[parents].detach());assert scene.count==20000
    with torch.no_grad():scene.log_scales[:2].fill_(math.log(1.8))
    torch.cuda.synchronize();start=time.perf_counter();opt.zero_grad(set_to_none=True);l=loss(renderer(scene,v.kr,v.kd,v.dr,v.dd));l.backward();torch.cuda.synchronize()
    assert torch.isfinite(l) and all(p.grad is None or torch.isfinite(p.grad).all() for p in scene.parameters())
    first=time.perf_counter()-start;timings=[]
    for _ in range(3):
        opt.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter();ll=loss(renderer(scene,v.kr,v.kd,v.dr,v.dd));ll.backward();torch.cuda.synchronize();timings.append(time.perf_counter()-start)
    report.update(stress20k_first_seconds=first,stress20k_seconds=sorted(timings)[1],stress20k_peak_gib=torch.cuda.max_memory_allocated()/2**30,source=base.__file__,passed=True)
    path=Path('outputs/paper_reproduction')/('check_'+mode+'.json');path.write_text(json.dumps(report,indent=2));print('CHECK_PASSED',json.dumps(report),flush=True)


if __name__=='__main__':check()
