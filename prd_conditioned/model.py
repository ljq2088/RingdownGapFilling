import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from prd_variable.model import VariableDGF
from prd_physical.common import N
from prd_strategy.latent import WholeWindowAdapter
from .settings import START,OUT

def phase_features(low):
    a=torch.log1p(low[:,:,0].clamp_min(0)); angle=low[:,:,1]
    z=torch.stack((a,a*torch.cos(angle),a*torch.sin(angle)),dim=2)
    return F.adaptive_avg_pool2d(z.reshape(-1,3,16,264),(4,8)).reshape(len(low),-1)

def context_features(u,p,noise_rms):
    f=torch.fft.rfft(u);g=torch.fft.rfft(p)
    f=f.clone();g=g.clone();f[...,0]=f[...,-1]=0;g[...,0]=g[...,-1]=0
    x=torch.fft.irfft(f,n=N);y=torch.fft.irfft(g,n=N)
    rms=x.square().mean(-1).sqrt().clamp_min(1e-6)
    expected=noise_rms.to(u.device).clamp_min(1e-6)
    logratio=torch.log(rms/expected).clamp(-8,8)
    frac=torch.log1p(expected/rms).clamp_max(8)
    bands=[]
    for lo,hi in ((1,5),(5,17),(17,65),(65,528)):
        energy=(2*f[...,lo:hi].abs().square().sum(-1)/N**2).sqrt()/rms
        bands.append(energy)
    residual=torch.log1p((x-y).square().mean(-1).sqrt()/expected).clamp_max(8)
    pred=torch.log1p(y.square().mean(-1).sqrt()/expected).clamp_max(8)
    return torch.cat((logratio,frac,*bands,residual,pred),dim=1)

def packet(u,p,mask,low,basis,noise_rms,sign=None):
    f=torch.fft.rfft(p); e=torch.zeros_like(f);e[...,0]=f[...,0];e[...,-1]=f[...,-1]
    ends=torch.fft.irfft(e,n=N); c=(p-ends).flatten(1)@basis;obs=u.flatten(1)@basis
    rms=u.square().mean((1,2)).sqrt();s=rms.clamp_min(1e-4)[:,None]*np.sqrt(2*N)
    start=mask.argmax(-1).to(u.dtype)/N;length=mask.sum(-1)/N
    cr=u.square().mean(-1).sqrt()/rms[:,None].clamp_min(1e-4)
    extras=torch.cat((start[:,None],(start+length)[:,None],length[:,None],cr,c.norm(dim=-1,keepdim=True)/s),dim=1)
    return dict(features=torch.cat((c/s,obs/s,extras),dim=1),coefficients=c,scale=s,context=context_features(u,p,noise_rms),phase=phase_features(low),endpoint_dc=f[...,0].real/N,endpoint_ny=f[...,-1].real/N,start=start,length=length,sign=torch.ones(len(u),device=u.device) if sign is None else sign)

class ResidualHead(nn.Module):
    def __init__(self,variant):
        super().__init__();self.variant=variant
        self.input=nn.Linear(470,192)
        self.blocks=nn.ModuleList([nn.Sequential(nn.LayerNorm(192),nn.Linear(192,192),nn.GELU(),nn.Dropout(.05),nn.Linear(192,192)) for _ in range(2)])
        self.head=nn.Linear(192,128);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,v):
        context=v['context'] if self.variant!='normalized_control' else torch.zeros_like(v['context'])
        phase=v['phase'] if self.variant!='noise_context' else torch.zeros_like(v['phase'])
        z=F.gelu(self.input(torch.cat((v['features'],context,phase),dim=1)))
        for b in self.blocks:z=z+.25*b(z)
        return v['coefficients']+.1*v['scale']*torch.tanh(self.head(z))

def reconstruct(head,v,basis):
    c=head(v);parity=torch.arange(N,device=c.device).remainder(2).mul(-2).add(1)
    endpoint=v['endpoint_dc'][...,None]+v['endpoint_ny'][...,None]*parity
    return (c@basis.T).reshape(-1,2,N)+endpoint

class ConditionedDGF(nn.Module):
    def __init__(self,variant):
        super().__init__();self.backbone=VariableDGF()
        self.backbone.load_state_dict(torch.load(START,map_location='cpu',weights_only=True)['state_dict'])
        self.register_buffer('noise_rms',torch.tensor(np.load(OUT/'noise_calibration.npz')['rms'],dtype=torch.float32))
        self.head=ResidualHead(variant)
    def input_bands(self,u):return self.backbone.input_bands(u)
    def forward(self,u,q,mask,bands=None,q_low=None):
        p=self.backbone(u,q,mask,bands,q_low)
        v=packet(u,p,mask,q_low,self.backbone.adapter.basis,self.noise_rms)
        return reconstruct(self.head,v,self.backbone.adapter.basis)
