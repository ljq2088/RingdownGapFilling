"""One physically valid noise/sign augmentation of 4096 training rows, frozen backbone."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'2')
os.environ.setdefault('NUMBA_NUM_THREADS','1')
import time,json
import h5py,numpy as np,torch
from ssqueezepy import cwt
from prd_physical.common import Preprocess,write
from prd_physical.prepare import PARENT,crop_meta
from prd_frequency.model import FrequencyDGF
from prd_frequency.features import features as low_features
from prd_physical.resources import available_gpu_bytes
from .settings import *
from .augment import orthogonal_noise_refresh,prepare_observation

def status(stage,**kw):
 v=dict(stage=stage,pid=os.getpid(),updated_unix=time.time(),**kw);write(OUT/'augmented_status.json',v);print(json.dumps(v),flush=True)
def main():
 if (OUT/'augmented_training.npz').exists():return
 assert json.loads((OUT/'augmentation_gates.json').read_text())['passed']
 torch.set_num_threads(2);device=os.environ.get('STRATEGY_DEVICE','cpu')
 if device=='cuda':torch.cuda.set_per_process_memory_fraction(GPU_FRACTION)
 pre=Preprocess(np.load(CAL)['psd'],'white');rng=np.random.default_rng(SEED+1)
 noise=[]
 with h5py.File(PARENT/'train.h5','r') as f:
  for i in range(PRED_TRAIN_N):
   start,_=crop_meta(f['parameters'][i],int(f['seeds'][i]));noise.append(f['noise'][i,:,start:start+1056])
 truth=np.load(CACHE/'train_truth.npy',mmap_mode='r')[:PRED_TRAIN_N];mask=np.load(CACHE/'train_mask.npy',mmap_mode='r')[:PRED_TRAIN_N]
 mixed=orthogonal_noise_refresh(np.stack(noise).astype(float),rng);sign=rng.choice([-1.,1.],PRED_TRAIN_N);u,target=prepare_observation(truth,mixed,mask,pre,sign)
 model=FrequencyDGF('frequency_adapter').to(device).eval();model.load_state_dict(torch.load(OUT/'frequency_snapshot.pth',map_location=device,weights_only=True)['state_dict']);pred=[];t=time.monotonic();last=0
 with torch.inference_mode():
  for lo in range(0,PRED_TRAIN_N,BATCH):
   hi=min(lo+BATCH,PRED_TRAIN_N);batch=u[lo:hi]
   scale=np.sqrt(np.mean(batch.astype(float)**2,axis=-1)).clip(1e-8)
   normalized=(batch/scale.astype('f4')[...,None]).reshape(-1,1056)
   z,_=cwt(normalized,'morlet',scales=np.linspace(1,16,16));z=torch.from_numpy(z.astype('complex64'))
   q=torch.stack([z.abs(),z.angle()],dim=1)[...,::4].reshape(len(batch),2,2,16,264)
   low=low_features(batch)
   if device=='cuda' and available_gpu_bytes()[0]<4.25*2**30:
    torch.cuda.empty_cache()
    while available_gpu_bytes()[0]<4.25*2**30:time.sleep(5)
   x=torch.from_numpy(batch).to(device);bands=model.input_bands(x)
   y=model(x,q.to(device),torch.from_numpy(np.array(mask[lo:hi],copy=True)).to(device),bands,torch.from_numpy(low).to(device)).cpu().numpy();assert np.isfinite(y).all();pred.append(y)
   if time.monotonic()-last>30 or hi==PRED_TRAIN_N:status('predicting_augmented_training',completed=hi,total=PRED_TRAIN_N,elapsed_seconds=time.monotonic()-t,device=device,peak_mb=(torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else 0));last=time.monotonic()
 np.savez_compressed(OUT/'augmented_training.npz',input=u,target=target,prediction=np.concatenate(pred),mask=mask,source_indices=np.arange(PRED_TRAIN_N),sign=sign)
 status('completed',n=PRED_TRAIN_N,all_features_recomputed=True,physical_noise_second_moment_preserved=True,test_read=False)
if __name__=='__main__':main()
