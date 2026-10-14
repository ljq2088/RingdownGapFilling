import json,numpy as np,torch
from torch import nn
from torch.nn import functional as F
from prd_physical.common import Preprocess
from .settings import OUT,CAL
from .common import analytic

class Dense(nn.Module):
    def __init__(self,nout,zero=True):
        super().__init__();self.input=nn.Linear(470,384)
        self.blocks=nn.ModuleList([nn.Sequential(nn.LayerNorm(384),nn.Linear(384,384),nn.GELU(),nn.Linear(384,384)) for _ in range(3)])
        self.output=nn.Linear(384,nout)
        if zero:nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,x):
        x=F.gelu(self.input(x))
        for block in self.blocks:x=x+.25*block(x)
        return self.output(x)

class ConvBlock(nn.Module):
    def __init__(self,ci,co):
        super().__init__();self.layers=nn.Sequential(nn.Conv1d(ci,co,5,padding=2),nn.GroupNorm(8,co),nn.GELU(),nn.Conv1d(co,co,5,padding=2),nn.GroupNorm(8,co),nn.GELU())
    def forward(self,x):return self.layers(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__();self.e0=ConvBlock(12,32);self.e1=ConvBlock(32,64);self.e2=ConvBlock(64,96);self.e3=ConvBlock(96,128)
        self.context=nn.Sequential(nn.Linear(470,192),nn.GELU(),nn.Linear(192,256))
        self.d2=ConvBlock(224,96);self.d1=ConvBlock(160,64);self.d0=ConvBlock(96,32);self.out=nn.Conv1d(32,2,1)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    def forward(self,x,features):
        a=self.e0(x);b=self.e1(F.avg_pool1d(a,2));c=self.e2(F.avg_pool1d(b,2));d=self.e3(F.avg_pool1d(c,2))
        g,bias=self.context(features).chunk(2,-1);d=d*(1+.1*torch.tanh(g[...,None]))+.1*bias[...,None]
        d=self.d2(torch.cat((F.interpolate(d,size=c.shape[-1],mode='linear',align_corners=False),c),1))
        d=self.d1(torch.cat((F.interpolate(d,size=b.shape[-1],mode='linear',align_corners=False),b),1))
        d=self.d0(torch.cat((F.interpolate(d,size=a.shape[-1],mode='linear',align_corners=False),a),1))
        return .1*self.out(d)

class TailModel(nn.Module):
    def __init__(self,variant):
        super().__init__();self.variant=variant
        pre=Preprocess(np.load(CAL)['psd'],'white')
        self.register_buffer('transfer',torch.tensor(pre.asd/pre.sigma.mean(),dtype=torch.float32))
        self.register_buffer('basis',torch.tensor(np.load(OUT/'basis.npy'),dtype=torch.float32))
        self.register_buffer('time',torch.arange(1056,dtype=torch.float32)*.5)
        if variant=='tail_basis':self.head=Dense(132)
        elif variant=='tail_unet':self.head=UNet()
        elif variant=='qnm_tail':
            self.head=Dense(3);self.strength=nn.Parameter(torch.zeros(()))
            table=np.load(OUT/'qnm_table.npz');self.register_buffer('f_table',torch.tensor(table['frequency'],dtype=torch.float32));self.register_buffer('tau_table',torch.tensor(table['tau'],dtype=torch.float32))
            self.stats=json.loads((OUT/'parameter_normalization.json').read_text())
        else:raise ValueError(variant)
    def physical(self,x):return torch.fft.irfft(torch.fft.rfft(x)*self.transfer,n=1056)
    def white(self,x):return torch.fft.irfft(torch.fft.rfft(x)/self.transfer,n=1056)
    def forward(self,v):
        p=v['prediction'];aux={}
        if self.variant=='tail_basis':
            x=self.head(v['features']);delta=(.1*v['scale']*torch.tanh(x[:,:128]))@self.basis.T
            dc=.1*v['scale']/np.sqrt(2112)*torch.tanh(x[:,128:130]);ny=.1*v['scale']/np.sqrt(2112)*torch.tanh(x[:,130:132])
            parity=1-2*torch.arange(1056,device=p.device).remainder(2)
            return p+delta.reshape_as(p)+dc[...,None]+ny[...,None]*parity,aux
        phys=self.physical(p);scale=phys.square().mean((1,2),keepdim=True).sqrt().clamp_min(1e-5)
        if self.variant=='tail_unet':
            obs=v['obs'];o=self.physical(obs);rms=obs.square().mean((1,2),keepdim=True).sqrt().clamp_min(1e-4)
            env=analytic(phys).abs()/scale
            inputs=torch.cat((obs/rms,p/rms,phys/scale,o/o.square().mean((1,2),keepdim=True).sqrt().clamp_min(1e-5),env,v['mask'][:,None],self.time[None,None].expand(len(p),1,-1)/528),1)
            residual=self.head(inputs,v['features'])*scale
            f=torch.fft.rfft(residual);freq=torch.fft.rfftfreq(1056,.5,device=p.device);window=(.5+.5*torch.cos(np.pi*((freq-.3)/.1).clamp(0,1)))
            residual=torch.fft.irfft(f*window,n=1056)
            return p+self.white(residual),aux
        x=self.head(v['features']);logmass=(self.stats['log_mass_mean']+self.stats['log_mass_std']*x[:,0]).clamp(np.log(1e5),np.log(5e7));spin=.4+.4*torch.sigmoid(x[:,1])
        aux['parameters']=torch.stack((logmass,spin),-1)
        loc=((spin-.4)/.0005).clamp(0,799.999);ix=loc.floor().long();mix=(loc-ix)[:,None]
        f=(self.f_table[ix]*(1-mix)+self.f_table[ix+1]*mix)/(torch.exp(logmass)[:,None]/1e6)
        tau=(self.tau_table[ix]*(1-mix)+self.tau_table[ix+1]*mix)*(torch.exp(logmass)[:,None]/1e6)
        anchor=(v['peak']+64).clamp(max=496);elapsed=(self.time[None]-anchor[:,None]).clamp_min(0)
        angle=2*np.pi*elapsed[:,:,None]*f[:,None];damp=torch.exp(-elapsed[:,:,None]/tau[:,None])
        design=torch.stack((damp*torch.cos(angle),damp*torch.sin(angle)),-1).flatten(-2)
        width=torch.minimum(3*tau[:,0],torch.full_like(tau[:,0],220.))
        weight=((self.time[None]>=anchor[:,None])&(self.time[None]<=anchor[:,None]+width[:,None])).to(p.dtype)*torch.exp(-.5*elapsed/tau[:,0,None])
        # The small positive ridge stabilizes nearly degenerate damped modes.
        B=design.double();wt=weight.double();y=(phys/scale).double()
        gram=torch.einsum('bni,bn,bnj->bij',B,wt,B);ridge=gram.diagonal(dim1=-2,dim2=-1).mean(-1)*1e-4+1e-8
        rhs=torch.einsum('bni,bn,bcn->bic',B,wt,y)
        coef=torch.linalg.solve(gram+ridge[:,None,None]*torch.eye(8,device=p.device,dtype=torch.float64),rhs)
        curve=torch.einsum('bni,bic->bcn',B,coef).to(p.dtype)*scale
        residual=((curve-phys).square()*weight[:,None]).sum((1,2))/(phys.square()*weight[:,None]).sum((1,2)).clamp_min(1e-10)
        gate=self.strength.clamp(0,1)*torch.sigmoid(x[:,2])*torch.exp(-3*residual.detach())*(v['peak']<460)
        blend=(elapsed/48).clamp(0,1);blend=blend.square()*(3-2*blend)
        return p+self.white((curve-phys)*(gate[:,None,None]*blend[:,None])),aux
    def clip(self):
        if self.variant=='qnm_tail':
            with torch.no_grad():self.strength.clamp_(0,1)
