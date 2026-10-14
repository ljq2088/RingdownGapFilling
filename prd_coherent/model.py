import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from prd_conditioned.model import ConditionedDGF,context_features
from prd_physical.common import N,Preprocess
from .settings import OUT,START,CAL

def interior(x):
    f=torch.fft.rfft(x);f=f.clone();f[...,0]=f[...,-1]=0
    return torch.fft.irfft(f,n=N)

class GapOperator(nn.Module):
    """L = W G W^-1, using the same physical-before-whitening gap convention."""
    def __init__(self):
        super().__init__();pre=Preprocess(np.load(CAL)['psd'],'white')
        self.register_buffer('transfer',torch.tensor(pre.asd/pre.sigma.mean(),dtype=torch.float32))
    def physical_mask(self,x,mask):
        physical=torch.fft.irfft(torch.fft.rfft(x)*self.transfer,n=N)
        shape=(len(mask),)+(1,)*(x.ndim-2)+(N,)
        physical=physical*(1-mask.reshape(shape))
        return torch.fft.irfft(torch.fft.rfft(physical)/self.transfer,n=N)
    def forward(self,x,mask):return interior(self.physical_mask(x,mask))
    def adjoint(self,x,mask):
        x=interior(x)
        x=torch.fft.irfft(torch.fft.rfft(x)/self.transfer,n=N)*(1-mask[:,None])
        return torch.fft.irfft(torch.fft.rfft(x)*self.transfer,n=N)

class DataConsistency(nn.Module):
    def __init__(self,basis,noise_rms):
        super().__init__();self.op=GapOperator()
        self.register_buffer('basis',basis.detach().clone())
        self.register_buffer('noise_rms',noise_rms.detach().clone())
        self.register_buffer('gram',torch.tensor(np.load(OUT/'mask_gram.npy'),dtype=torch.float32))
        self.context=nn.Sequential(nn.Linear(16,32),nn.GELU(),nn.Linear(32,2))
        nn.init.zeros_(self.context[-1].weight);nn.init.zeros_(self.context[-1].bias)
        with torch.no_grad():self.context[-1].bias[0]=3.
        self.strength=nn.Parameter(torch.zeros(()))
    def forward(self,u,p,mask):
        length=mask.sum(-1).round().long();start=mask.argmax(-1)
        original=start+(length-66)//2
        index=(original-132)*21+(length-64)//2
        assert bool(((index>=0)&(index<len(self.gram))).all())
        context=context_features(u,p,self.noise_rms)
        values=self.context(context)
        lam=F.softplus(values[:,0])+.05
        gate=self.strength.clamp(0,.5)*torch.sigmoid(values[:,1])
        residual=interior(u)-self.op(p,mask)
        rhs=self.op.adjoint(residual,mask).flatten(1)@self.basis
        H=self.gram[index]+lam[:,None,None]*torch.eye(128,device=u.device)
        L=torch.linalg.cholesky(H)
        delta=torch.cholesky_solve(rhs[...,None],L).squeeze(-1)
        return p+(gate[:,None]*delta@self.basis.T).reshape_as(p)

class CoherentDGF(nn.Module):
    def __init__(self,variant):
        super().__init__();self.variant=variant
        self.net=ConditionedDGF('noise_phase')
        self.net.load_state_dict(torch.load(START,map_location='cpu',weights_only=True)['state_dict'])
        self.dc=DataConsistency(self.net.backbone.adapter.basis,self.net.noise_rms) if variant=='data_consistency' else None
    def input_bands(self,x):return self.net.input_bands(x)
    def forward(self,u,q,mask,bands=None,q_low=None):
        p=self.net(u,q,mask,bands,q_low)
        return self.dc(u,p,mask) if self.dc is not None else p
    def training_modes(self):
        self.train()
        # Keep the inherited encoder dropout deterministic while adapting weights.
        self.net.backbone.dgf.backbone.dgf.base.encoder.eval()
    def clip_constraints(self):
        if self.dc is not None:
            with torch.no_grad():self.dc.strength.clamp_(0,.5)

def optimizer(model):
    for p in model.parameters():p.requires_grad_(False)
    n=model.net;base=n.backbone.dgf.backbone.dgf.base
    modules=[(n.head,2e-5,'coefficients'),(n.backbone.adapter.coefficient_adapter,5e-6,'basis_adapter'),
      (n.backbone.dgf.frequency,4e-6,'physical_frequency'),(n.backbone.dgf.backbone.joint,3e-6,'joint_channels'),
      (n.backbone.dgf.backbone.dgf.refiner,3e-6,'mask_refiner'),(base.decoder,3e-6,'waveform_decoder')]
    if model.variant=='encoder_adaptation':
        modules.extend([(base.encoder.QtranTimeMixer,2e-7,'time_frequency_encoder'),
                        (base.encoder.totalembedding,5e-7,'temporal_embedding')])
        for j,layer in enumerate(base.encoder.transformer.layers[-4:]):
            # The inherited allocated MultiheadAttention tensors are inactive.
            for name in ('feedforward','norm1','norm2'):
                modules.append((getattr(layer,name),1e-6,f'encoder_tail_{j}_{name}'))
    if model.dc is not None:modules.append((model.dc,3e-4,'data_consistency'))
    groups=[];seen=set()
    for module,lr,name in modules:
        params=[p for p in module.parameters() if id(p) not in seen]
        if not params:continue
        for p in params:p.requires_grad_(True);seen.add(id(p))
        groups.append(dict(params=params,lr=lr,name=name))
    return torch.optim.AdamW(groups,weight_decay=2e-4)
