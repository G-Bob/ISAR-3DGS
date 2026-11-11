from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class RadarVGGTDiagnostics:
    loss: float
    candidates: int
    selected: int
    evidence_mean: float

class RadarVGGTInitializer(nn.Module):
    def __init__(self, dim=64, heads=4, layers=2):
        super().__init__()
        self.embed=nn.Linear(14,dim)
        def block(): return nn.TransformerEncoderLayer(dim,heads,dim*2,dropout=0.0,batch_first=True,norm_first=True)
        self.frame=nn.TransformerEncoder(block(),layers)
        self.global_=nn.TransformerEncoder(block(),layers)
        self.depth=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,1),nn.Tanh())
        self.confidence=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,1))
    def forward(self,x):
        local=self.frame(self.embed(x))
        context=self.global_(local.mean(1)[None])[0]
        fused=local+context[:,None]
        return self.depth(fused).squeeze(-1),self.confidence(fused).squeeze(-1)

def _tokens(views,grid,device):
    axis=torch.linspace(-1,1,grid,device=device); rr,cc=torch.meshgrid(axis,axis,indexing='ij')
    uv=torch.stack((rr.flatten(),cc.flatten()),-1); tokens=[]; bases=[]; normals=[]
    for v in views:
        image=v.image.to(device=device,dtype=torch.float32)[None,None]
        amp=F.adaptive_avg_pool2d(image,(grid,grid)).flatten()
        kr=v.kr.to(device=device,dtype=torch.float32); kd=v.kd.to(device=device,dtype=torch.float32)
        normal=v.normal.to(device=device,dtype=torch.float32)
        geom=torch.cat((kr,kd,normal))[None].expand(grid*grid,-1)
        spacing=torch.tensor([v.dr,v.dd],device=device)[None].expand(grid*grid,-1)
        tokens.append(torch.cat((amp[:,None],uv,geom,spacing),-1))
        h,w=v.image.shape[-2:]
        r=rr.flatten()*(h-1)*0.5*float(v.dr); c=cc.flatten()*(w-1)*0.5*float(v.dd)
        bases.append(r[:,None]*kr+c[:,None]*kd); normals.append(normal[None].expand(grid*grid,-1))
    return torch.stack(tokens),torch.stack(bases),torch.stack(normals)

def _evidence(points,views,device):
    samples=[]
    for v in views:
        image=v.image.to(device=device,dtype=torch.float32)[None,None]; h,w=v.image.shape[-2:]
        kr=v.kr.to(device); kd=v.kd.to(device)
        row=(points@kr)/(max(h-1,1)*0.5*float(v.dr)); col=(points@kd)/(max(w-1,1)*0.5*float(v.dd))
        grid=torch.stack((col,row),-1).view(1,-1,1,2)
        samples.append(F.grid_sample(image,grid,mode='bilinear',padding_mode='zeros',align_corners=True).flatten())
    values=torch.stack(samples)
    return 0.5*values.mean(0)+0.5*values.median(0).values

def radar_vggt_surface(count,bbox,seed,views,view_count=16,patch_grid=16,steps=120,learning_rate=2e-3,device='cuda'):
    if not views: raise ValueError('radar_vggt_surface requires ISAR views')
    torch.manual_seed(seed); dev=torch.device(device if device.startswith('cuda') and torch.cuda.is_available() else 'cpu')
    ids=np.linspace(0,len(views)-1,min(view_count,len(views)),dtype=int); chosen=[views[int(i)] for i in ids]
    tokens,bases,normals=_tokens(chosen,patch_grid,dev); model=RadarVGGTInitializer().to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=1e-4); limit=0.08*bbox
    loss=torch.tensor(float('nan'),device=dev)
    for _ in range(max(steps,0)):
        depth,logits=model(tokens); points=bases+depth[...,None]*limit*normals
        evidence=_evidence(points.flatten(0,1),chosen,dev).view_as(depth)
        target=(evidence.detach()/evidence.detach().amax().clamp_min(1e-6)).clamp(0,1)
        loss=F.binary_cross_entropy_with_logits(logits,target)-0.05*evidence.mean()+0.01*depth.square().mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad():
        depth,logits=model(tokens); points=bases+depth[...,None]*limit*normals
        evidence=_evidence(points.flatten(0,1),chosen,dev).view_as(depth); score=torch.sigmoid(logits)*evidence.clamp_min(0)
        fp=points.flatten(0,1); fn=normals.flatten(0,1); order=torch.argsort(score.flatten(),descending=True)
        if order.numel()<count: order=order.repeat((count+order.numel()-1)//order.numel())
        order=order[:count]
        p=fp[order].clamp(-bbox,bbox).cpu().numpy().astype(np.float32)
        n=F.normalize(fn[order],dim=-1).cpu().numpy().astype(np.float32)
        info=RadarVGGTDiagnostics(float(loss.cpu()),int(fp.shape[0]),count,float(evidence.mean().cpu()))
    return p,n,info
