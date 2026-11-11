"""Exact ordered RGB alpha accumulation, 3sigma; streaming image-domain tiles."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def mul(a,b):return a*b


@tr.jit
def forward_kernel(M,Q,A,C,O,N:tl.constexpr,H:tl.constexpr,W:tl.constexpr,G:tl.constexpr=32,P:tl.constexpr=32):
    pix=tl.program_id(0)*P+tl.arange(0,P);valid=pix<H*W
    r=(pix//W).to(tl.float32);c=(pix%W).to(tl.float32)
    trans=tl.full((P,),1.,tl.float32);s0=tl.full((P,),0.,tl.float32);s1=s0;s2=s0
    for start in range(0,tl.cdiv(N,G)):
        g=start*G+tl.arange(0,G);vg=g<N
        dr=r[None,:]-tl.load(M+2*g,vg,0)[:,None];dc=c[None,:]-tl.load(M+2*g+1,vg,0)[:,None]
        q0=tl.load(Q+3*g,vg,0);q1=tl.load(Q+3*g+1,vg,0);q2=tl.load(Q+3*g+2,vg,0)
        z=q0[:,None]*dr*dr+2*q1[:,None]*dr*dc+q2[:,None]*dc*dc
        a=tl.minimum(tl.load(A+g,vg,0)[:,None]*tl.exp(-.5*z),.99)
        a=tl.where(vg[:,None]&valid[None,:]&(z<=9),a,0.)
        cumulative=tl.associative_scan(1-a,0,mul);before=cumulative/(1-a)
        weight=trans[None,:]*before*a
        s0+=tl.sum(weight*tl.load(C+3*g,vg,0)[:,None],0)
        s1+=tl.sum(weight*tl.load(C+3*g+1,vg,0)[:,None],0)
        s2+=tl.sum(weight*tl.load(C+3*g+2,vg,0)[:,None],0)
        trans*=tl.sum(tl.where(tl.arange(0,G)[:,None]==G-1,cumulative,0.),0)
    tl.store(O+3*pix,s0,valid);tl.store(O+3*pix+1,s1,valid);tl.store(O+3*pix+2,s2,valid)


@tr.jit
def backward_kernel(M,Q,A,C,O,D,DM,DQ,DA,DC,N:tl.constexpr,H:tl.constexpr,W:tl.constexpr,G:tl.constexpr=32,P:tl.constexpr=32):
    pix=tl.program_id(0)*P+tl.arange(0,P);vp=pix<H*W
    r=(pix//W).to(tl.float32);c=(pix%W).to(tl.float32)
    y0=tl.load(O+3*pix,vp,0);y1=tl.load(O+3*pix+1,vp,0);y2=tl.load(O+3*pix+2,vp,0)
    d0=tl.load(D+3*pix,vp,0);d1=tl.load(D+3*pix+1,vp,0);d2=tl.load(D+3*pix+2,vp,0)
    trans=tl.full((P,),1.,tl.float32);s0=tl.full((P,),0.,tl.float32);s1=s0;s2=s0
    for start in range(0,tl.cdiv(N,G)):
        g=start*G+tl.arange(0,G);vg=g<N
        dr=r[None,:]-tl.load(M+2*g,vg,0)[:,None];dc=c[None,:]-tl.load(M+2*g+1,vg,0)[:,None]
        q0=tl.load(Q+3*g,vg,0);q1=tl.load(Q+3*g+1,vg,0);q2=tl.load(Q+3*g+2,vg,0)
        z=q0[:,None]*dr*dr+2*q1[:,None]*dr*dc+q2[:,None]*dc*dc
        valid=vg[:,None]&vp[None,:]&(z<=9)
        kernel=tl.where(valid,tl.exp(-.5*z),0.);unclamped=tl.load(A+g,vg,0)[:,None]*kernel
        a=tl.minimum(unclamped,.99);cum=tl.associative_scan(1-a,0,mul);t=trans[None,:]*cum/(1-a)
        c0=tl.load(C+3*g,vg,0);c1=tl.load(C+3*g+1,vg,0);c2=tl.load(C+3*g+2,vg,0)
        v0=t*a*c0[:,None];v1=t*a*c1[:,None];v2=t*a*c2[:,None]
        prefix0=tl.cumsum(v0,0)-v0+s0[None,:];prefix1=tl.cumsum(v1,0)-v1+s1[None,:];prefix2=tl.cumsum(v2,0)-v2+s2[None,:]
        grad=((t*c0[:,None]-y0[None,:]+prefix0)*d0[None,:]+(t*c1[:,None]-y1[None,:]+prefix1)*d1[None,:]+(t*c2[:,None]-y2[None,:]+prefix2)*d2[None,:])/(1-a)
        grad=tl.where(valid&(unclamped<.99),grad,0.)
        zgrad=grad*unclamped
        tl.atomic_add(DA+g,tl.sum(grad*kernel,1),vg)
        tl.atomic_add(DM+2*g,tl.sum(zgrad*(q0[:,None]*dr+q1[:,None]*dc),1),vg)
        tl.atomic_add(DM+2*g+1,tl.sum(zgrad*(q1[:,None]*dr+q2[:,None]*dc),1),vg)
        tl.atomic_add(DQ+3*g,tl.sum(-.5*zgrad*dr*dr,1),vg)
        tl.atomic_add(DQ+3*g+1,tl.sum(-zgrad*dr*dc,1),vg)
        tl.atomic_add(DQ+3*g+2,tl.sum(-.5*zgrad*dc*dc,1),vg)
        tl.atomic_add(DC+3*g,tl.sum(t*a*d0[None,:],1),vg)
        tl.atomic_add(DC+3*g+1,tl.sum(t*a*d1[None,:],1),vg)
        tl.atomic_add(DC+3*g+2,tl.sum(t*a*d2[None,:],1),vg)
        s0+=tl.sum(v0,0);s1+=tl.sum(v1,0);s2+=tl.sum(v2,0)
        trans*=tl.sum(tl.where(tl.arange(0,G)[:,None]==G-1,cum,0.),0)


class Alpha(torch.autograd.Function):
    @staticmethod
    def forward(ctx,means,conic,alpha,color,h,w):
        means=means.contiguous();conic=conic.contiguous();alpha=alpha.contiguous();color=color.contiguous()
        out=means.new_empty(h,w,3);forward_kernel[(tr.cdiv(h*w,32),)](means,conic,alpha,color,out,len(means),h,w)
        ctx.save_for_backward(means,conic,alpha,color,out);ctx.hw=(h,w);return out
    @staticmethod
    def backward(ctx,dy):
        m,q,a,c,o=ctx.saved_tensors;dm=torch.zeros_like(m);dq=torch.zeros_like(q);da=torch.zeros_like(a);dc=torch.zeros_like(c);h,w=ctx.hw
        backward_kernel[(tr.cdiv(h*w,32),)](m,q,a,c,o,dy.contiguous(),dm,dq,da,dc,len(m),h,w)
        return dm,dq,da,dc,None,None


def rasterize(means,conic,alpha,color,h,w):return Alpha.apply(means,conic,alpha,color,h,w)
