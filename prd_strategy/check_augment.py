import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('NUMBA_NUM_THREADS','1')
import numpy as np,h5py,torch
from prd_physical.common import Preprocess,write
from prd_physical.prepare import PARENT,crop_meta
from model.QTranTimeMixerMod import QTransformModule
from prd_frequency.features import features
from .settings import *
from .augment import orthogonal_noise_refresh,prepare_observation,changed_crop

def main():
 cal=np.load(CAL);pre=Preprocess(cal['psd'],'white');noise=[];recrop=[];rng=np.random.default_rng(SEED)
 with h5py.File(PARENT/'train.h5','r') as f:
  for i in range(512):
   par=f['parameters'][i];seed=int(f['seeds'][i]);start,m=crop_meta(par,seed);noise.append(f['noise'][i,:,start:start+1056])
  q=QTransformModule()
  for i in [0,11999,12000,23456,40000,59999]:
   h,n,m,start=changed_crop(f,i,2041000000+i)
   assert m.sum()==66 and np.isfinite(h).all() and np.isfinite(n).all()
   u,v=prepare_observation(h[None],n[None],m[None],pre)
   assert np.array_equal(u,pre.encode((h.astype(np.float64)*(1-m.astype(np.float64))+n.astype(np.float64))[None]))
   negative,_=prepare_observation(h[None],n[None],m[None],pre,[-1]);assert np.array_equal(negative,-u)
   scale=np.sqrt(np.mean(u.astype(float)**2,axis=-1)).clip(1e-8)
   original_q=np.stack([q(torch.from_numpy(row/sc)).numpy() for row,sc in zip(u[0],scale[0])]);low_q=features(u)
   assert np.isfinite(original_q).all() and np.isfinite(low_q).all()
   recrop.append(dict(index=i,start=start,gap_samples=int(m.sum()),observation_and_global_sign_reproduced=True,both_cwt_banks_recomputed=True))
 noise=np.stack(noise).astype(float);refreshed=orthogonal_noise_refresh(noise,rng)
 F=np.fft.rfft(noise);G=np.fft.rfft(refreshed)
 power=(abs(F)**2).sum(0);newpower=(abs(G)**2).sum(0);psd_error=float(np.max(abs(newpower-power)/np.maximum(power,np.finfo(float).tiny)))
 ix=np.unique(np.linspace(1,527,24).astype(int));a=np.concatenate([F[...,ix].real.reshape(len(F),-1),F[...,ix].imag.reshape(len(F),-1)],axis=1);b=np.concatenate([G[...,ix].real.reshape(len(G),-1),G[...,ix].imag.reshape(len(G),-1)],axis=1)
 cov=a.T@a;changed=b.T@b;cov_error=float(np.linalg.norm(changed-cov)/np.linalg.norm(cov));assert psd_error<1e-12 and cov_error<1e-12
 assert float(np.linalg.norm(refreshed-noise)/np.linalg.norm(noise))>.1
 result=dict(passed=True,noise_bank_rows=512,maximum_empirical_periodogram_relative_error=psd_error,joint_AE_cross_frequency_uncentered_second_moment_relative_error=cov_error,noise_changed=True,mask_semantics='G*h+n; noise retained in the gap',noise_units='fractional frequency',recrop_rows=recrop,sign_symmetry='simultaneous signal+noise sign, corresponding to polarization psi+pi/2 and zero-mean Gaussian noise',proof_scope='orthogonal row rotation exactly preserves the noise-bank uncentered second moment; iid same-covariance Gaussian rows retain their joint Gaussian law',test_read=False)
 write(OUT/'augmentation_gates.json',result);print(result,flush=True)
if __name__=='__main__':main()
