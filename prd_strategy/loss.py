"""Matched physical-space components; balanced weights are a declared probe."""
import torch
from prd_physical.common import N
class StrategyObjective:
 def __init__(self,pre,cal,balanced,auxiliary_weights=None):
  self.balanced=balanced;self.auxiliary_weights=auxiliary_weights;self.w=torch.tensor(pre.weights());self.transfer=torch.tensor(pre.asd/pre.sigma.mean(),dtype=torch.float32);self.hann=torch.tensor(__import__('numpy').hanning(N),dtype=torch.float32)
  hw=pre.sigma.mean()**2/cal['hann_psd'];hw=hw.copy();hw[:,0]=hw[:,-1]=0;self.hw=torch.tensor(hw/hw.mean(),dtype=torch.float32)
 def __call__(self,p,h,mask):
  P,H=torch.fft.rfft(p),torch.fft.rfft(h);tt=(H.abs().square()*self.w).sum((1,2));pp=(P.abs().square()*self.w).sum((1,2));cross=((H*P.conj()).real*self.w).sum((1,2))
  mm=1-cross/(tt*pp).clamp_min(1e-24).sqrt();spec=((P-H).abs().square()*self.w).sum((1,2))/tt.clamp_min(1e-12)
  a=torch.fft.irfft(P*self.transfer,n=N);b=torch.fft.irfft(H*self.transfer,n=N);e=(a-b).square();den=b.square().sum((1,2)).clamp_min(1e-12);time=(e.sum((1,2))+2*(e*mask[:,None]).sum((1,2)))/den
  A,B=torch.fft.rfft(a*self.hann),torch.fft.rfft(b*self.hann);t=(B.abs().square()*self.hw).sum((1,2));q=(A.abs().square()*self.hw).sum((1,2));c=((B*A.conj()).real*self.hw).sum((1,2));rel=((A-B).abs().square()*self.hw).sum((1,2))/t.clamp_min(1e-12);hm=1-c/(t*q).clamp_min(1e-24).sqrt();amp=2*(torch.sqrt(1+rel)-1)
  tw,hw=(.05,.02) if self.balanced else (.5,.15)
  if self.auxiliary_weights is not None:tw,hw=self.auxiliary_weights
  return (mm+.5*spec+tw*time+hw*(hm+.25*amp)).mean()
