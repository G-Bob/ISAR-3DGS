"""Exact additive Gaussian rasterizer for the Pei BSAT adaptation; no alpha blend."""
import torch
import triton
import triton.language as tl


@triton.jit
def forward_kernel(M,C,A,IDS,O,H:tl.constexpr,W:tl.constexpr,R:tl.constexpr,B:tl.constexpr):
    ix=tl.load(IDS+tl.program_id(0));k=tl.arange(0,B);side=2*R+1
    mr=tl.load(M+ix*2);mc=tl.load(M+ix*2+1)
    pr=tl.floor(mr).to(tl.int32)+k//side-R;pc=tl.floor(mc).to(tl.int32)+k%side-R
    dy=pr-mr;dx=pc-mc;c0=tl.load(C+ix*3);c1=tl.load(C+ix*3+1);c2=tl.load(C+ix*3+2)
    maha=c0*dy*dy+2*c1*dy*dx+c2*dx*dx
    valid=(k<side*side)&(pr>=0)&(pr<H)&(pc>=0)&(pc<W)&(maha<=9)
    value=tl.exp(-.5*maha)*tl.load(A+ix)
    tl.atomic_add(O+pr*W+pc,value,mask=valid)


@triton.jit
def backward_kernel(M,C,A,IDS,G,GM,GC,GA,H:tl.constexpr,W:tl.constexpr,R:tl.constexpr,B:tl.constexpr):
    ix=tl.load(IDS+tl.program_id(0));k=tl.arange(0,B);side=2*R+1
    mr=tl.load(M+ix*2);mc=tl.load(M+ix*2+1)
    pr=tl.floor(mr).to(tl.int32)+k//side-R;pc=tl.floor(mc).to(tl.int32)+k%side-R
    dy=pr-mr;dx=pc-mc;c0=tl.load(C+ix*3);c1=tl.load(C+ix*3+1);c2=tl.load(C+ix*3+2)
    maha=c0*dy*dy+2*c1*dy*dx+c2*dx*dx
    valid=(k<side*side)&(pr>=0)&(pr<H)&(pc>=0)&(pc<W)&(maha<=9)
    g=tl.load(G+pr*W+pc,mask=valid,other=0);ker=tl.exp(-.5*maha);v=g*ker*tl.load(A+ix)
    tl.store(GM+ix*2,tl.sum(v*(c0*dy+c1*dx),0));tl.store(GM+ix*2+1,tl.sum(v*(c1*dy+c2*dx),0))
    tl.store(GC+ix*3,tl.sum(-.5*v*dy*dy,0));tl.store(GC+ix*3+1,tl.sum(-v*dy*dx,0));tl.store(GC+ix*3+2,tl.sum(-.5*v*dx*dx,0))
    tl.store(GA+ix,tl.sum(g*ker,0))


class Additive(torch.autograd.Function):
    @staticmethod
    def forward(ctx,means,conics,weights,radii,height,width):
        means=means.contiguous();conics=conics.contiguous();weights=weights.contiguous()
        assert means.is_cuda and means.dtype==torch.float32
        inside=(means[:,0]+radii>=0)&(means[:,0]-radii<height)&(means[:,1]+radii>=0)&(means[:,1]-radii<width)
        groups=[];out=means.new_zeros(height*width)
        for r in torch.unique(radii[inside]).tolist():
            ids=torch.where(inside&(radii==r))[0].contiguous();b=triton.next_power_of_2((2*r+1)**2)
            warps=4 if b<=1024 else 8 if b<=8192 else 32
            if r>32:
                wide_forward[(len(ids),triton.cdiv(height*width,256))](means,conics,weights,ids,out,height,width,256)
            else:
                forward_kernel[(len(ids),)](means,conics,weights,ids,out,height,width,r,b,num_warps=warps)
            groups.append((r,ids,b,warps))
        ctx.save_for_backward(means,conics,weights);ctx.groups=groups;ctx.shape=(height,width)
        return out.reshape(height,width)
    @staticmethod
    def backward(ctx,gradient):
        means,conics,weights=ctx.saved_tensors;gm=torch.zeros_like(means);gc=torch.zeros_like(conics);ga=torch.zeros_like(weights)
        g=gradient.contiguous();h,w=ctx.shape
        for r,ids,b,warps in ctx.groups:
            if r>32:
                wide_backward[(len(ids),triton.cdiv(h*w,256))](means,conics,weights,ids,g,gm,gc,ga,h,w,256)
            else:
                backward_kernel[(len(ids),)](means,conics,weights,ids,g,gm,gc,ga,h,w,r,b,num_warps=warps)
        return gm,gc,ga,None,None,None


def rasterize(means,conics,weights,radii,height,width):
    return Additive.apply(means,conics,weights,radii,height,width)


@triton.jit
def wide_forward(M,C,A,IDS,O,H:tl.constexpr,W:tl.constexpr,B:tl.constexpr):
    ix=tl.load(IDS+tl.program_id(0));k=tl.program_id(1)*B+tl.arange(0,B)
    dy=k//W-tl.load(M+ix*2);dx=k%W-tl.load(M+ix*2+1)
    c0=tl.load(C+ix*3);c1=tl.load(C+ix*3+1);c2=tl.load(C+ix*3+2)
    maha=c0*dy*dy+2*c1*dy*dx+c2*dx*dx
    tl.atomic_add(O+k,tl.exp(-.5*maha)*tl.load(A+ix),mask=(k<H*W)&(maha<=9))


@triton.jit
def wide_backward(M,C,A,IDS,G,GM,GC,GA,H:tl.constexpr,W:tl.constexpr,B:tl.constexpr):
    ix=tl.load(IDS+tl.program_id(0));k=tl.program_id(1)*B+tl.arange(0,B)
    dy=k//W-tl.load(M+ix*2);dx=k%W-tl.load(M+ix*2+1)
    c0=tl.load(C+ix*3);c1=tl.load(C+ix*3+1);c2=tl.load(C+ix*3+2)
    maha=c0*dy*dy+2*c1*dy*dx+c2*dx*dx
    g=tl.load(G+k,mask=(k<H*W)&(maha<=9),other=0);ker=tl.exp(-.5*maha);v=g*ker*tl.load(A+ix)
    tl.atomic_add(GM+ix*2,tl.sum(v*(c0*dy+c1*dx),0));tl.atomic_add(GM+ix*2+1,tl.sum(v*(c1*dy+c2*dx),0))
    tl.atomic_add(GC+ix*3,tl.sum(-.5*v*dy*dy,0));tl.atomic_add(GC+ix*3+1,tl.sum(-v*dy*dx,0));tl.atomic_add(GC+ix*3+2,tl.sum(-.5*v*dx*dx,0))
    tl.atomic_add(GA+ix,tl.sum(g*ker,0))

