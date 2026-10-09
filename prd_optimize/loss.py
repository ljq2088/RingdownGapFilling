"""Declared physical-space objective; auxiliary windows never change the main metric."""
import torch
from prd_physical.run import lossfn
from prd_physical.common import N
class Objective:
 def __init__(self,pre,cal,variant,device="cuda"):
  self.variant=variant
  self.w=torch.tensor(pre.weights(),device=device)
  self.transfer=torch.tensor(pre.asd/pre.sigma.mean(),device=device,dtype=torch.float32)
  self.hann=torch.hann_window(N,periodic=False,device=device)
  w=pre.sigma.mean()**2/cal["hann_psd"];w=w.copy();w[:,0]=w[:,-1]=0;w=w/w.mean()
  self.hw=torch.tensor(w,device=device,dtype=torch.float32)
 def __call__(self,p,h,m):
  basic=lossfn(p,h,m,self.w,self.transfer)
  if self.variant=="continuation":return basic
  a=torch.fft.irfft(torch.fft.rfft(p,dim=-1)*self.transfer,n=N,dim=-1)
  b=torch.fft.irfft(torch.fft.rfft(h,dim=-1)*self.transfer,n=N,dim=-1)
  P=torch.fft.rfft(a*self.hann,dim=-1);H=torch.fft.rfft(b*self.hann,dim=-1)
  tt=(H.abs().square()*self.hw).sum((-2,-1));pp=(P.abs().square()*self.hw).sum((-2,-1))
  cross=((H*P.conj()).real*self.hw).sum((-2,-1))
  mm=1-cross/(tt*pp).clamp_min(1e-24).sqrt()
  rel=((P-H).abs().square()*self.hw).sum((-2,-1))/tt.clamp_min(1e-12)
  # Smooth robust amplitude penalty includes every source; no clipping/filtering.
  amplitude=2*(torch.sqrt(1+rel)-1)
  return basic+.15*(mm+.25*amplitude).mean()
