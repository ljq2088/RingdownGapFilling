import numpy as np
import torch
from prd_physical.common import Preprocess,N
from .settings import CAL

class Objective:
    def __init__(self):
        pre=Preprocess(np.load(CAL)['psd'],'white')
        self.transfer=torch.tensor(pre.asd/pre.sigma.mean(),device='cuda',dtype=torch.float32)
        a=torch.zeros(N,device='cuda');a[0]=a[N//2]=1;a[1:N//2]=2;self.analytic=a
    def __call__(self,p,h,mask,epoch=5):
        P=torch.fft.rfft(p);H=torch.fft.rfft(h);a=P[...,1:-1];b=H[...,1:-1]
        tt=b.abs().square().sum((1,2)).clamp_min(1e-12);pp=a.abs().square().sum((1,2)).clamp_min(1e-12)
        mm=1-(a.conj()*b).real.sum((1,2))/(tt*pp).sqrt()
        relative=(a-b).abs().square().sum((1,2))/tt
        u=torch.fft.irfft(P*self.transfer,n=N);v=torch.fft.irfft(H*self.transfer,n=N)
        e=(u-v).square();physical=(e.sum((1,2))+2*(e*mask[:,None]).sum((1,2)))/v.square().sum((1,2)).clamp_min(1e-12)
        # Per-channel peak-phase supervision in physical coordinates.
        z=torch.fft.ifft(torch.fft.fft(v)*self.analytic)
        w=torch.fft.ifft(torch.fft.fft(u)*self.analytic)
        j=z.abs().argmax(-1,keepdim=True)
        z=z.gather(-1,j).squeeze(-1);w=w.gather(-1,j).squeeze(-1)
        denom=(w.abs().square()+1e-8*z.abs().square()).sqrt()*z.abs().clamp_min(1e-8)
        phase=1-(w*z.conj()).real/denom.clamp_min(1e-12)
        hc=b.abs().square().sum(-1);pc=a.abs().square().sum(-1)
        rho2=2/N*hc;reliability=rho2/(rho2+25.)
        cm=1-(a.conj()*b).real.sum(-1)/(hc*pc).clamp_min(1e-12).sqrt()
        phase=(phase*reliability).mean(-1);channel=(cm*reliability).mean(-1)
        loss=mm+.2*relative+.02*physical+.1*min(1.,epoch/5)*phase+.05*channel
        return loss.mean()
