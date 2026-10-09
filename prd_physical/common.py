"""Direct signed overlaps and invertible noise-only preprocessing."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
N=1056; FS=2.; SEED=2026100931

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()

def write(path,value):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def metric(h,p,psd=None):
 # Float64 protects physical fractional-frequency norms (~1e-40).
 h=np.asarray(h,np.float64);p=np.asarray(p,np.float64)
 if psd is None:a,b=h,p;w=1.
 else:
  a,b=np.fft.rfft(h,axis=-1),np.fft.rfft(p,axis=-1)
  w=1/np.asarray(psd,np.float64);w=w.copy();w[...,0]=w[...,-1]=0
 cross=(np.real(a*b.conj())*w).sum(axis=(-2,-1))
 aa=(abs(a)**2*w).sum(axis=(-2,-1));bb=(abs(b)**2*w).sum(axis=(-2,-1))
 if np.any(aa<=0):raise ValueError('zero target norm; abort without filtering')
 match=np.divide(cross,np.sqrt(aa*bb),out=np.zeros_like(cross),where=bb>0)
 return 1-np.clip(match,-1,1)

def summary(x):
 x=np.asarray(x);assert np.isfinite(x).all()
 return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),p95=float(np.quantile(x,.95)),max=float(x.max()))

class Preprocess:
 def __init__(self,psd,kind):
  self.psd=np.asarray(psd,np.float64);self.kind=kind
  self.sigma=np.sqrt((self.psd[:,1:-1].sum(-1)+.5*self.psd[:,-1])*FS/N)
  self.asd=np.sqrt(self.psd*FS/2)
  # Keep endpoints invertible even though the overlap excludes them.
  self.asd[:,0]=self.asd[:,1];self.asd[:,-1]=self.asd[:,-2]
  assert np.isfinite(self.asd).all() and (self.asd>0).all()
 def encode(self,x):
  x=np.asarray(x,np.float64)
  y=x/self.sigma[:,None] if self.kind=='raw' else np.fft.irfft(np.fft.rfft(x,axis=-1)/self.asd,n=N,axis=-1)
  return y.astype(np.float32)
 def decode(self,y):
  y=np.asarray(y,np.float64)
  return y*self.sigma[:,None] if self.kind=='raw' else np.fft.irfft(np.fft.rfft(y,axis=-1)*self.asd,n=N,axis=-1)
 def weights(self):
  # Work-space weights reproduce the direct PHYSICAL joint A/E overlap.
  transfer=self.sigma[:,None] if self.kind=='raw' else self.asd
  w=transfer**2/self.psd;w=w.copy();w[:,0]=w[:,-1]=0
  return (w/w.mean()).astype(np.float32)
