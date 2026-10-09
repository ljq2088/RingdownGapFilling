"""Numerical convention tests; no train/validation/test target access."""
import numpy as np,torch
from .common import metric,Preprocess,N,FS,write
from .run import lossfn,OUT

def main():
 t=np.arange(N);h=np.stack([np.cos(2*np.pi*13*t/N),np.cos(2*np.pi*21*t/N)])[None]
 phase=.37;p=np.stack([np.cos(2*np.pi*13*t/N+phase),np.cos(2*np.pi*21*t/N+phase)])[None]
 assert np.allclose(metric(h,p),1-np.cos(phase),atol=1e-12)
 assert np.allclose(metric(h,h*9),0,atol=1e-12)
 assert np.allclose(metric(h,-h),2,atol=1e-12)
 assert np.allclose(metric(h,p,np.ones((2,N//2+1))),metric(h,p),atol=1e-12)
 psd=np.load(OUT/'calibration_psd.npz')['psd']
 assert np.allclose(metric(h,p,psd*1e-5),metric(h,p,psd),atol=1e-12)
 # A low-noise erroneous bin is weighted MORE, and can increase mismatch.
 a=np.zeros((1,2,N));a[:,0]=np.cos(2*np.pi*13*t/N)
 b=a.copy();b[:,0]+=np.cos(2*np.pi*21*t/N)*.1
 flat=np.ones((2,N//2+1));low=flat.copy();low[0,21]=.01
 assert metric(a,b,low)[0]>metric(a,b,flat)[0]
 rng=np.random.default_rng(83);physical_h=rng.normal(size=(5,2,N))*1e-20;physical_p=physical_h+rng.normal(size=(5,2,N))*1e-21
 masks=np.zeros((5,N),np.float32);masks[:,198:264]=1;losses={}
 for kind in ['raw','white']:
  pre=Preprocess(psd,kind);transfer=pre.sigma[:,None] if kind=='raw' else pre.asd
  losses[kind]=float(lossfn(torch.from_numpy(pre.encode(physical_p)),torch.from_numpy(pre.encode(physical_h)),torch.from_numpy(masks),torch.tensor(pre.weights()),torch.tensor(transfer/pre.sigma.mean(),dtype=torch.float32)))
 assert abs(losses['raw']-losses['white'])<1e-6,losses
 write(OUT/'metric_gates.json',dict(passed=True,phase_expected=1-np.cos(phase),positive_amplitude_invariance=True,signed_range=[0,2],white_parseval=True,global_psd_scaling_invariance=True,low_psd_erroneous_bin_increases_mismatch=True,matched_physical_losses=losses,physical_loss_delta=abs(losses['raw']-losses['white'])))
 print(losses)

if __name__=='__main__':main()
