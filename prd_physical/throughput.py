"""Compute-only batch sizing; no validation or test access."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('NUMBA_NUM_THREADS','1')
import time,json
import torch,numpy as np
from .model import PhysicalDGF
from .run import OUT,BASE,lossfn
from .common import SEED,N,write

def main():
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.20);torch.manual_seed(SEED)
 rows=[]
 for b in [16,32,64,128]:
  model=PhysicalDGF(BASE).cuda();opt=torch.optim.AdamW(model.parameters(),lr=1e-5)
  u=torch.randn(b,2,N,device='cuda');h=u*.8;q=torch.randn(b,2,2,16,264,device='cuda');mask=torch.zeros(b,N,device='cuda');mask[:,198:264]=1;w=torch.ones(2,N//2+1,device='cuda');w[:,0]=w[:,-1]=0
  times=[];torch.cuda.reset_peak_memory_stats()
  for k in range(6):
   t=time.time();model.train();opt.zero_grad(set_to_none=True);loss=lossfn(model(u,q,mask),h,mask,w);loss.backward();opt.step();torch.cuda.synchronize()
   times.append(time.time()-t)
  row=dict(batch_sources=b,median_step_seconds=float(np.median(times[2:])),sources_per_second=b/float(np.median(times[2:])),peak_gpu_mb=torch.cuda.max_memory_allocated()/2**20)
  rows.append(row);print(json.dumps(row),flush=True)
  del model,opt,u,h,q,mask,w,loss;torch.cuda.empty_cache()
 write(OUT/'throughput_probe.json',dict(rows=rows,purpose='compute-only sizing, not training evidence'))

if __name__=='__main__':main()
