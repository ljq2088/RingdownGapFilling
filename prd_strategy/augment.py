"""Exact pairwise Gaussian-noise rotations and physical recropping of training rows."""
import numpy as np
from prd_physical.common import Preprocess
from prd_physical.prepare import crop_meta

def orthogonal_noise_refresh(noise,rng):
 noise=np.asarray(noise);count=len(noise)
 if count%2:raise ValueError('pairwise refresh requires an even-sized noise bank')
 order=rng.permutation(count);a,b=order[0::2],order[1::2];angle=rng.uniform(0,2*np.pi,len(a));c=np.cos(angle)[:,None,None];s=np.sin(angle)[:,None,None]
 out=np.empty_like(noise,dtype=np.float64)
 out[a]=c*noise[a]+s*noise[b];out[b]=-s*noise[a]+c*noise[b]
 return out

def prepare_observation(truth,noise,mask,pre,sign=None):
 h=np.asarray(truth,float);n=np.asarray(noise,float);m=np.asarray(mask,float)
 if sign is not None:h=h*np.asarray(sign)[:,None,None];n=n*np.asarray(sign)[:,None,None]
 return pre.encode(h*(1-m[:,None,:])+n),pre.encode(h)

def changed_crop(parent,index,augmentation_seed):
 par=parent['parameters'][index];start,mask=crop_meta(par,int(augmentation_seed));return parent['truth'][index,:,start:start+1056],parent['noise'][index,:,start:start+1056],mask,start
