"""Shared copy-on-write memory maps: no outcome filtering and no target-derived features."""
import time,json,os
import numpy as np,torch
from prd_physical.common import Preprocess
from .settings import OUT,OLD,KEYS,BATCH,GPU_FRACTION,MIN_GPU_FREE
from prd_physical.resources import available_gpu_bytes

def load_maps(split):
 return {k:torch.from_numpy(np.load(OUT/"cache"/(split+"_"+k+".npy"),mmap_mode="c")) for k in KEYS}
def cache_prefix(a,count):
 count=min(count,len(a["input"]));needed=sum(a[k][:count].numel()*a[k].element_size() for k in KEYS)
 free,total=available_gpu_bytes()
 if free-needed-3*2**30<MIN_GPU_FREE or torch.cuda.memory_allocated()+needed+3*2**30>total*GPU_FRACTION:
  return dict(rows=0,bytes=0,reason="preserve activation reserve and free VRAM")
 before=torch.cuda.get_rng_state().clone()
 a["_gpu_cache"]={k:a[k][:count].cuda() for k in KEYS};a["_cache_rows"]=count
 assert torch.equal(before,torch.cuda.get_rng_state())
 return dict(rows=count,bytes=needed,rng_unchanged=True)
def offload_cache(a):
 a.pop("_gpu_cache",None);a.pop("_cache_rows",None)
def batches(a,order):
 for lo in range(0,len(order),BATCH):
  ix=np.asarray(order[lo:lo+BATCH]);cpu_ix=torch.as_tensor(ix,dtype=torch.long);cache=a.get("_gpu_cache")
  if cache is None:
   yield {k:a[k].index_select(0,cpu_ix).to("cuda") for k in KEYS}
  else:
   cached=ix<a["_cache_rows"];pos=np.flatnonzero(cached);other=np.flatnonzero(~cached);result={}
   ci=torch.as_tensor(ix[cached],device="cuda",dtype=torch.long);cp=torch.as_tensor(pos,device="cuda",dtype=torch.long);op=torch.as_tensor(other,device="cuda",dtype=torch.long)
   for k in KEYS:
    out=torch.empty((len(ix),*a[k].shape[1:]),dtype=a[k].dtype,device="cuda")
    if len(pos):out.index_copy_(0,cp,cache[k].index_select(0,ci))
    if len(other):out.index_copy_(0,op,a[k].index_select(0,torch.as_tensor(ix[~cached],dtype=torch.long)).cuda())
    result[k]=out
   yield result
def physical_truth(split):return np.load(OUT/"cache"/(split+"_truth.npy"),mmap_mode="r").astype(np.float64)
def predict(model,a,pre):
 model.eval();p=[]
 with torch.no_grad():
  for v in batches(a,np.arange(len(a["input"]))):
   out=model(v["input"],v["q"],v["mask"],v["bands"]).cpu().numpy()
   if not np.isfinite(out).all():raise ValueError("nonfinite prediction")
   p.append(pre.decode(out))
 return np.concatenate(p)
