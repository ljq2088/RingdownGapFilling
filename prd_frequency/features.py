"""Observed-input CWT features, fixed before training, with physical Hz coverage."""
import numpy as np
from ssqueezepy import cwt
from prd_physical.common import N,FS
FREQUENCIES=np.geomspace(.25,.002,16)
SCALES=(13.4*FS/(2*np.pi*FREQUENCIES)).astype(np.float32)
def features(u):
 u=np.asarray(u);assert u.ndim==3 and u.shape[1:]==(2,N)
 scale=np.sqrt(np.mean(u.astype(np.float64)**2,axis=-1)).clip(1e-8)
 normalized=(u/scale[...,None]).reshape(-1,N)
 z,_=cwt(normalized,'morlet',scales=SCALES)
 out=np.stack([np.abs(z),np.angle(z)],axis=1)[...,::4]
 return out.reshape(len(u),2,2,16,264).astype(np.float32)
