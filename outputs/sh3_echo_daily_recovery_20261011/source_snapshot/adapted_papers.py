"""Disclosed BSAT adaptations; no native-paper/PHARE reproduction claim."""
import math, os, sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
import pei_bsat_wide as base
from radar_gs.geometry import projection_coordinates, projection_jacobian

MODE=os.environ.get('PAPER_MODE','pei_mask')


def rgb_target(x):
    # Frozen invertible display embedding; scalar comparison uses channel0.
    return torch.stack((x,x.square(),(x+1e-6).sqrt()-.001),-1)


def attenuation(scene,kr,kd):
    # Eq13 continuum approximation: integrate exp(-mahalanobis) over the
    # whole LOS-orthogonal plane; 3sigma tail factor is explicit. This is
    # integrated global attenuation, not per-ray local occlusion.
    n=torch.linalg.cross(kr,kd);n=n/n.norm();b=torch.stack((kd/kd.norm(),n))
    cov=b@scene.covariance()@b.T
    mass=scene.support_probability()*math.pi*torch.linalg.det(cov).clamp_min(1e-16).sqrt()*(1-math.exp(-9))
    # Existing projection convention: increasing depth kr dot means is far.
    order=(scene.means@kr).argsort(stable=True);before=torch.cumsum(mass[order],0)-mass[order]
    result=torch.empty_like(before);result[order]=torch.exp(-before);return result


class Scene(base.Scene):
    def __init__(self,means,bbox=6.):
        super().__init__(means,bbox)
        if MODE=='isargs':
            gen=torch.Generator(device='cpu').manual_seed(4232)
            self.imag_dc=nn.Parameter(torch.randn(len(means),1,generator=gen).to(means)*.01)
            self.imag_rest=nn.Parameter(torch.randn(len(means),15,generator=gen).to(means)*.01)
        elif MODE.startswith('rns'):
            self.sh_dc=nn.Parameter(self.sh_dc.detach()[:,:,None].expand(-1,-1,3).clone())
            self.sh_rest=nn.Parameter(self.sh_rest.detach()[:,:,None].expand(-1,-1,3).clone())
            if MODE=='rns_rsu':
                self.rsu_band=nn.Parameter(means.new_tensor(1.))
                self.rsu_beta=nn.Parameter(means.new_tensor(-2.))
    def complex_amplitude(self,kr):
        k=(self.degree+1)**2;b=base.sh3(kr/kr.norm())[:k]
        return torch.complex(.5+(torch.cat((self.sh_dc,self.sh_rest),-1)[:,:k]*b).sum(-1),(torch.cat((self.imag_dc,self.imag_rest),-1)[:,:k]*b).sum(-1))
    def amplitude(self,kr):
        if not MODE.startswith('rns'):return super().amplitude(kr)
        k=(self.degree+1)**2;b=base.sh3(kr/kr.norm())[:k]
        color=(.5+(torch.cat((self.sh_dc,self.sh_rest),1)[:,:k,:]*b[None,:,None]).sum(1)).clamp_min(0.)
        if MODE=='rns_rsu':
            bandwidth=torch.nn.functional.softplus(self.rsu_band)
            coords=self.means.new_tensor([-1.,0.,1.])*bandwidth
            gain=1+.25*self.rsu_beta.sigmoid()*torch.sinc(coords)*torch.exp(-.5*coords.square())
            color=color*gain[None,:]
        return color


class Renderer(nn.Module):
    def __init__(self,height,width,*args,**kwargs):super().__init__();self.height=height;self.width=width;self.use_cuda=True
    def forward(self,scene,kr,kd,dr,dd,**kwargs):
        kr=kr.to(scene.means);kd=kd.to(scene.means)
        row,col,depth=projection_coordinates(scene.means,kr,kd,self.height,self.width,dr,dd)
        centers=torch.stack((row,col),-1)
        if centers.requires_grad:centers.retain_grad()
        p=projection_jacobian(kr,kd,dr,dd);cov=p@scene.covariance()@p.T+torch.eye(2,device=kr.device)[None]*.3
        inv=torch.linalg.inv(cov);conic=torch.stack((inv[:,0,0],inv[:,0,1],inv[:,1,1]),-1)
        alpha=scene.support_probability()
        if MODE=='isargs':
            from pei_linear_cuda_wide import rasterize
            disc=(cov[:,0,0]-cov[:,1,1]).square()+4*cov[:,0,1].square()
            radius=torch.ceil(3*((cov[:,0,0]+cov[:,1,1]+disc.sqrt())*.5).detach().sqrt()).long().clamp_min(1)
            radius=2**torch.ceil(torch.log2(radius.float())).long()
            z=scene.complex_amplitude(kr)*alpha*attenuation(scene,kr,kd)
            re=rasterize(centers,conic,z.real.contiguous(),radius,self.height,self.width)
            im=rasterize(centers,conic,z.imag.contiguous(),radius,self.height,self.width)
            image=torch.sqrt(re.square()+im.square()+1e-12)
            return SimpleNamespace(image=image,raw_image=image,projected_centers=centers)
        from alpha_cuda import rasterize
        order=depth.argsort(stable=True);color=scene.amplitude(kr)
        rgb=rasterize(centers[order],conic[order],alpha[order],color[order],self.height,self.width)
        return SimpleNamespace(image=rgb[:,:,0],raw_image=rgb[:,:,0],rgb=rgb,projected_centers=centers)


def install():
    if MODE=='pei_mask':
        # Reuse exact M19 mask+guard, no tracking, no hidden GT input.
        sys.path.insert(0,str(Path('outputs').resolve()));import round6_fast_worker as r6
        r6.GUARD=True
    else:
        base.Scene=Scene;base.LinearRenderer=Renderer
        if MODE=='isargs':base.NAMES+=['imag_dc','imag_rest'];base.LRS+=[.0025,.000125]
    original_make=base.make_optimizer
    if MODE=='rns_rsu':
        def make(scene):
            opt=original_make(scene)
            opt.add_param_group({'params':[scene.rsu_band],'lr':.001,'name':'rsu_band'})
            opt.add_param_group({'params':[scene.rsu_beta],'lr':.001,'name':'rsu_beta'})
            return opt
        base.make_optimizer=make
    base.PAPER_MODE=MODE
    source=Path(base.__file__).read_text()
    source=source.replace("cfg['implementation']='Pei equation-based BSAT budget adaptation; NOT official/native paper result'","cfg['implementation']='Disclosed BSAT adaptation: '+PAPER_MODE+'; not official/native reproduction';cfg['paper_mode']=PAPER_MODE")
    if MODE=='pei_mask':
        source=source.replace("loss=.8*(pred-target).abs().mean()+.2*(1-ssim(pred,target))","loss=mask_guard_loss(pred,target,.1,.2)[0]")
        base.mask_guard_loss=r6.masked_loss
    elif MODE.startswith('rns'):
        source=source.replace("loss=.8*(pred-target).abs().mean()+.2*(1-ssim(pred,target))","rgb=rendered.rgb;rgbt=rgb_target(target);loss=.8*(rgb-rgbt).abs().mean()+.2*sum(1-ssim(rgb[:,:,j],rgbt[:,:,j]) for j in range(3))/3+1e-4*scene.sh_rest.square().mean()+1e-3*(scene.log_scales-scene.log_scales.mean(-1,keepdim=True)).square().mean()")
    source=source.replace("        name=group['name'];old=getattr(scene,name)","        name=group['name']\n        if name.startswith('rsu_'):continue\n        old=getattr(scene,name)")
    # Compile only the two affected functions, retaining installed subclasses.
    import ast
    tree=ast.parse(source)
    for n in tree.body:
        if isinstance(n,ast.FunctionDef) and n.name in ['train','change_topology']:
            exec(compile(ast.Module(body=[n],type_ignores=[]),base.__file__,'exec'),base.__dict__)
    base.rgb_target=rgb_target


if __name__=='__main__':
    install();parser=base.argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--out',required=True);parser.add_argument('--init',default='train_points');parser.add_argument('--iters',type=int,default=12000)
    base.train(parser.parse_args())
