"""Legacy reproduction and controlled finite-gradient transfer-learning probe."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('NUMBA_NUM_THREADS','1')
import time,json
import numpy as np
import torch
import h5py
from .run import legacy_gate,lossfn,OUT,BASE,status
from .common import Preprocess,N,FS,SEED,write
from .prepare import crop_meta,PARENT
from .physics import Simulator
from .model import PhysicalDGF

def main():
 OUT.mkdir(exist_ok=True);torch.set_num_threads(2);torch.manual_seed(SEED)
 torch.cuda.set_per_process_memory_fraction(.20)
 legacy_gate()
 from model.QTranTimeMixerMod import QTransformModule
 q=QTransformModule();sim=Simulator();sums=np.zeros((2,N//2+1))
 for i in range(64):
  noise=sim.noise(2027100000+i)[:,6000:6000+N];sums+=abs(np.fft.rfft(noise))**2/(FS*N)
 psd=sums/64;psd[:,1:-1]*=2
 hs=[];xs=[];ms=[]
 with h5py.File(PARENT/'train.h5','r') as f:
  for i in range(16):
   s,m=crop_meta(f['parameters'][i],int(f['seeds'][i]));h=f['truth'][i,:,s:s+N];n=f['noise'][i,:,s:s+N]
   hs.append(h);xs.append(h*(1-m)+n);ms.append(m)
 results={}
 for kind in ['raw','white']:
  pre=Preprocess(psd,kind);u=pre.encode(np.stack(xs));v=pre.encode(np.stack(hs));scales=np.sqrt((u.astype(float)**2).mean(-1)).clip(1e-8)
  qt=np.stack([torch.stack([q(torch.from_numpy(row/sc)) for row,sc in zip(a,b)]).numpy() for a,b in zip(u,scales)])
  u=torch.from_numpy(u).cuda();v=torch.from_numpy(v).cuda();qt=torch.from_numpy(qt).cuda();m=torch.from_numpy(np.stack(ms)).cuda();w=torch.tensor(pre.weights(),device='cuda');transfer=pre.sigma[:,None] if kind=='raw' else pre.asd;time_transfer=torch.tensor(transfer/pre.sigma.mean(),device='cuda',dtype=torch.float32)
  torch.manual_seed(SEED);model=PhysicalDGF(BASE).cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-5)
  losses=[];times=[];torch.cuda.reset_peak_memory_stats();model.eval()
  with torch.no_grad():initial=float(lossfn(model(u,qt,m),v,m,w,time_transfer))
  for step in range(25):
   t=time.time();model.train();opt.zero_grad(set_to_none=True);p=model(u,qt,m);loss=lossfn(p,v,m,w,time_transfer);assert torch.isfinite(loss)
   loss.backward();assert model.base.encoder.totalembedding.token_embedding.linear.weight.grad is not None
   torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();torch.cuda.synchronize();losses.append(float(loss.detach()));times.append(time.time()-t)
  model.eval()
  with torch.no_grad():final=float(lossfn(model(u,qt,m),v,m,w,time_transfer))
  if final>=initial:raise ValueError((kind,'small-batch probe did not learn',initial,final))
  result=dict(initial_loss=initial,final_loss=final,steps=25,median_step_seconds=float(np.median(times[5:])),batch_sources=16,total_parameters=sum(p.numel() for p in model.parameters()),peak_gpu_mb=torch.cuda.max_memory_allocated()/2**20,finite_gradients=True,purpose='16-example engineering probe, not validation or a performance claim')
  results[kind]=result;print(json.dumps({kind:result}),flush=True)
  del model,opt,u,v,qt,m,w;torch.cuda.empty_cache()
 write(OUT/'training_probe.json',dict(passed=True,results=results))

if __name__=='__main__':main()
