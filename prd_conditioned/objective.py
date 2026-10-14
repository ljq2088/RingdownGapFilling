import numpy as np,torch
from prd_physical.common import N,Preprocess
from .settings import CAL
class Objective:
    def __init__(self):
        pre=Preprocess(np.load(CAL)['psd'],'white')
        self.transfer=torch.tensor(pre.asd/pre.sigma.mean(),dtype=torch.float32,device='cuda')
    def __call__(self,p,h,mask):
        P=torch.fft.rfft(p);H=torch.fft.rfft(h)
        a=P[...,1:-1];b=H[...,1:-1]
        tt=b.abs().square().sum((1,2)).clamp_min(1e-12);pp=a.abs().square().sum((1,2)).clamp_min(1e-12)
        cross=(a.conj()*b).real.sum((1,2))
        mm=1-cross/(tt*pp).sqrt();relative=(a-b).abs().square().sum((1,2))/tt
        u=torch.fft.irfft(P*self.transfer,n=N);v=torch.fft.irfft(H*self.transfer,n=N)
        e=(u-v).square();den=v.square().sum((1,2)).clamp_min(1e-12)
        physical=(e.sum((1,2))+2*(e*mask[:,None]).sum((1,2)))/den
        return (mm+.2*relative+.02*physical).mean()
