"""Global train-derived low-rank DGF output adapter; inference uses observation/prediction/mask."""
import numpy as np,torch
from torch import nn
from prd_physical.common import N
RANK=128
class CoefficientAdapter(nn.Module):
 def __init__(self,rank=RANK):
  super().__init__();self.encoder=nn.Sequential(nn.Linear(2*rank+6,128),nn.GELU(),nn.Linear(128,128),nn.GELU());self.head=nn.Linear(128,rank)
  nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
 def forward(self,features,coefficients,scale):return coefficients+.2*scale*self.head(self.encoder(features))

def coordinates(observation,prediction,mask,basis):
 observation=np.asarray(observation,float);prediction=np.asarray(prediction,float);mask=np.asarray(mask,float)
 F=np.fft.rfft(prediction);ends=F.copy();ends[...,1:-1]=0;endpoint=np.fft.irfft(ends,n=N)
 p=(prediction-endpoint).reshape(len(prediction),-1)@basis
 x=observation.reshape(len(observation),-1)@basis
 scale=np.sqrt((observation**2).mean((1,2))).clip(1e-4)[:,None]*np.sqrt(2*N)
 start=mask.argmax(-1)/N;length=mask.sum(-1)/N;end=start+length
 cr=np.sqrt((observation**2).mean(-1))/np.sqrt((observation**2).mean((1,2)))[:,None].clip(1e-4)
 pn=np.linalg.norm(p,axis=-1,keepdims=True)/scale
 extras=np.c_[start,end,length,cr,pn]
 features=np.concatenate([p/scale,x/scale,extras],axis=1)
 return dict(features=features.astype('f4'),coefficients=p.astype('f4'),scale=scale.astype('f4'),endpoint=endpoint.astype('f4'))

class WholeWindowAdapter(nn.Module):
 """Differentiable joint-A/E projection plus observation-conditioned coefficient residual.

 The basis is fitted on noise-free training directions. The zero head reproduces
 the fixed projection; DC and Nyquist coefficients of the backbone are retained.
 No source parameters or target-dependent alignment enter inference.
 """
 def __init__(self,basis):
  super().__init__()
  basis=torch.as_tensor(basis,dtype=torch.float32)
  if basis.ndim!=2 or basis.shape[0]!=2*N:raise ValueError('expected joint A/E reduced basis')
  self.register_buffer('basis',basis.clone());self.coefficient_adapter=CoefficientAdapter(basis.shape[1])
 def forward(self,observation,prediction,mask):
  f=torch.fft.rfft(prediction);ends=torch.zeros_like(f);ends[...,0]=f[...,0];ends[...,-1]=f[...,-1]
  endpoint=torch.fft.irfft(ends,n=N)
  p=(prediction-endpoint).flatten(1)@self.basis;x=observation.flatten(1)@self.basis
  rms=observation.square().mean((1,2)).sqrt();scale=rms.clamp_min(1e-4)[:,None]*np.sqrt(2*N)
  start=mask.argmax(-1).to(prediction.dtype)/N;length=mask.sum(-1)/N
  cr=observation.square().mean(-1).sqrt()/rms[:,None].clamp_min(1e-4)
  pn=p.norm(dim=-1,keepdim=True)/scale
  extras=torch.cat([start[:,None],(start+length)[:,None],length[:,None],cr,pn],dim=1)
  features=torch.cat([p/scale,x/scale,extras],dim=1)
  c=self.coefficient_adapter(features,p,scale)
  return (c@self.basis.T).reshape(-1,2,N)+endpoint
