"""Audit frozen parent data; create declared 528-s DGF crops, never read test targets."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('NUMBA_NUM_THREADS','1')
import json,zipfile,time,hashlib
from pathlib import Path
import numpy as np
import h5py
from .physics import Simulator,CONFIG,PARAM_NAMES,gap_mask
from .verify import verify
from .common import N,FS,SEED,sha,write,Preprocess,metric
ROOT=Path(__file__).resolve().parents[1]
PARENT=Path('/home/ljq/code/RingdownGapFilling/FillingGap/results_physical_retraining_20261009')
OUT=ROOT/'FillingGap/results_prd_physical_transfer_20261009'
COUNTS={'train':12000,'validation':1500,'test':6000}

def crop_meta(parameters,seed):
 # The supplied gap fixes the segment placement; no peak/target is passed to inference.
 _,g=gap_mask(int(parameters[13]),int(seed)+200000000)
 local=int(np.random.default_rng(int(seed)+300000000).integers(132,264))
 start=int(np.clip(g-local,0,32768-N));m=np.zeros(N,np.float32)
 m[g-start:g-start+66]=1
 if m.sum()!=66:raise ValueError('inadequate crop; abort, never exclude row')
 return start,m

def prepare_crops(split,count):
 if split=='test':raise RuntimeError('test targets remain sealed until checkpoint selection')
 path=OUT/(split+'_crops.h5')
 if path.exists():
  with h5py.File(path,'r') as f:assert f.attrs['completed']==count
  return
 tmp=path.with_suffix('.tmp')
 with h5py.File(PARENT/(split+'.h5'),'r') as src,h5py.File(tmp,'w') as dst:
  for k in ['truth','noise']:dst.create_dataset(k,shape=(count,2,N),dtype='f4',chunks=(32,2,N))
  dst.create_dataset('mask',shape=(count,N),dtype='f4',chunks=(32,N))
  dst.create_dataset('parameters',shape=(count,15),dtype='f8');dst.create_dataset('seeds',shape=(count,),dtype='i8')
  dst.create_dataset('crop_starts',shape=(count,),dtype='i4')
  for i in range(count):
   pars=src['parameters'][i];seed=int(src['seeds'][i]);s,m=crop_meta(pars,seed)
   dst['truth'][i]=src['truth'][i,:,s:s+N];dst['noise'][i]=src['noise'][i,:,s:s+N]
   dst['mask'][i]=m;dst['parameters'][i]=pars;dst['seeds'][i]=seed;dst['crop_starts'][i]=s
   if (i+1)%1000==0:print(json.dumps(dict(stage='prepare_crops',split=split,completed=i+1,total=count)),flush=True)
  dst.attrs['completed']=count;dst.attrs['observation']='G*h+n';dst.attrs['units']='fractional frequency'
 tmp.replace(path)

def qcache(split,kind):
 path=OUT/(split+'_'+kind+'_features.h5')
 if path.exists():
  with h5py.File(path,'r') as f:assert f.attrs['completed']==COUNTS[split]
  return
 import torch
 from model.QTranTimeMixerMod import QTransformModule
 q=QTransformModule();pre=Preprocess(np.load(OUT/'calibration_psd.npz')['psd'],kind)
 tmp=path.with_suffix('.tmp');count=COUNTS[split];t=time.time()
 with h5py.File(OUT/(split+'_crops.h5'),'r') as src,h5py.File(tmp,'w') as dst:
  dst.create_dataset('q',shape=(count,2,2,16,264),dtype='f4',chunks=(8,2,2,16,264))
  dst.create_dataset('input',shape=(count,2,N),dtype='f4',chunks=(32,2,N))
  dst.create_dataset('target',shape=(count,2,N),dtype='f4',chunks=(32,2,N))
  for i in range(count):
   h,n,m=src['truth'][i],src['noise'][i],src['mask'][i]
   x=h*(1-m)+n;u=pre.encode(x);v=pre.encode(h)
   # Identical observed-input RMS normalization for raw and white CWT features.
   scale=np.sqrt(np.mean(u.astype(float)**2,axis=-1)).clip(1e-8)
   qt=torch.stack([q(torch.from_numpy(row/sc)) for row,sc in zip(u,scale)]).numpy()
   assert np.isfinite(qt).all() and np.isfinite(u).all() and np.isfinite(v).all()
   dst['q'][i]=qt;dst['input'][i]=u;dst['target'][i]=v
   if (i+1)%1000==0:print(json.dumps(dict(stage='CWT_cache',split=split,kind=kind,completed=i+1,total=count,remaining_seconds=(count-i-1)*(time.time()-t)/(i+1))),flush=True)
  dst.attrs['completed']=count;dst.attrs['preprocessing']=kind
 tmp.replace(path)

def prepare():
 OUT.mkdir(exist_ok=True)
 if (OUT/'prepared.json').exists():return
 verify(OUT)
 sim=Simulator();checks=[]
 # Includes samples on both sides of the documented regenerated prefix.
 with h5py.File(PARENT/'train.h5','r') as f:
  for i in [0,511,7985,7986,11999,59999]:
   seed=int(f['seeds'][i]);h,n,p=sim.example(seed)
   for name,a,b in [('truth',h,f['truth'][i]),('noise',n,f['noise'][i]),('parameters',p,f['parameters'][i])]:
    assert np.array_equal(a,b),(i,name,'parent snapshot not reproduced')
   checks.append(dict(index=i,seed=seed,bitwise_reproduced=True))
 manifest={}
 for split,count in COUNTS.items():
  with h5py.File(PARENT/(split+'.h5'),'r') as f:
   assert f.attrs['completed']>=count
   seeds=f['seeds'][:count]
   manifest[split]=dict(parent_path=str(PARENT/(split+'.h5')),parent_rows=len(f['seeds']),rows_used=count,seed_min=int(seeds.min()),seed_max=int(seeds.max()),seeds_sha256=hashlib.sha256(seeds.tobytes()).hexdigest(),target_read=split!='test')
 for a,b in [('train','validation'),('train','test'),('validation','test')]:
  assert manifest[a]['seed_max']<manifest[b]['seed_min']
 protocol=dict(physics={**CONFIG,'observation':'G*h+n'},splits=manifest,crop_samples=N,crop_seconds=N/FS,gap_samples=66,crop_rule='segment placement from supplied gap start with seeded local gap start 132..263; parent gap population is near injected peak; no outcome exclusion',scope='finite 528-s two-channel segments; no claim of recovering the full longest ringdown',mask_semantics='1 suppresses signal; noise remains inside gap; not missing spacecraft samples',selection='unmaximized joint A/E mean PSD mismatch on validation only',test_policy='no test target read until both raw/white training checkpoints are selected; retain all 6000 rows',same_budget_variants=['raw','white'],cwt='Morlet magnitude and phase of input divided by observed-input channel RMS; no target feature',noise_psd='512 independent pure-noise crops, rectangular window; diagonal finite-crop metric, not full correlated likelihood',max_epochs=600,min_epochs=100,patience=100,min_delta=1e-5,batch_sources=16,seed=SEED,parent_train_attribute_note='early train config lacks numerical_ringup key; six stored rows including prefix boundary reproduce current frozen simulator bitwise',reproduction=checks)
 write(OUT/'protocol.json',protocol)
 if not (OUT/'calibration_psd.npz').exists():
  sums=np.zeros((2,N//2+1));hannsum=np.zeros_like(sums);hann=np.hanning(N);cal_seed=2027100000
  for i in range(512):
   noise=sim.noise(cal_seed+i)[:,6000:6000+N]
   sums+=abs(np.fft.rfft(noise,axis=-1))**2/(FS*N)
   hannsum+=abs(np.fft.rfft(noise*hann,axis=-1))**2/(FS*(hann*hann).sum())
  psd=sums/512;psd[:,1:-1]*=2;hpsd=hannsum/512;hpsd[:,1:-1]*=2
  assert np.isfinite(psd).all() and (psd>0).all()
  np.savez(OUT/'calibration_psd.npz',psd=psd,hann_psd=hpsd,frequency_hz=np.fft.rfftfreq(N,1/FS),seed_start=cal_seed,n_noise=512)
 psd=np.load(OUT/'calibration_psd.npz')['psd'];x=np.random.default_rng(SEED).normal(size=(8,2,N))*1e-20
 gates={}
 for kind in ['raw','white']:
  pre=Preprocess(psd,kind);rest=pre.decode(pre.encode(x));err=float(np.linalg.norm(rest-x)/np.linalg.norm(x));assert err<1e-6
  p=x*.9+np.random.default_rng(6).normal(size=x.shape)*1e-21
  physical=metric(x,p,psd);weights=pre.weights();inverse=np.full_like(weights,np.inf);np.divide(1,weights,out=inverse,where=weights>0);working=metric(pre.encode(x),pre.encode(p),inverse)
  assert np.max(abs(physical-working))<1e-7
  gates[kind]=dict(roundtrip_relative_l2=err,overlap_equivalence_max_error=float(np.max(abs(physical-working))))
 write(OUT/'preprocessing_gates.json',gates)
 for split in ['train','validation']:
  prepare_crops(split,COUNTS[split])
  for kind in ['raw','white']:qcache(split,kind)
 write(OUT/'prepared.json',dict(passed=True,protocol_sha256=sha(OUT/'protocol.json'),psd_sha256=sha(OUT/'calibration_psd.npz')))

if __name__=='__main__':prepare()
