import torch,numpy as np
from prd_physical.common import write
from .settings import *
from .data import load_maps,batches,cache_prefix,offload_cache
from .model import OptimizedDGF

def main():
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(GPU_FRACTION)
 a=load_maps("train");model=OptimizedDGF("joint_hann").cuda().eval()
 order=np.random.default_rng(SEED).choice(TRAIN_N,128,replace=False);order[:8]=[0,8191,8192,59999,1,12000,10000,7]
 original=next(batches(a,order));before=torch.cuda.get_rng_state().clone();info=cache_prefix(a,8192)
 assert info["rows"]==8192 and torch.equal(before,torch.cuda.get_rng_state())
 cached=next(batches(a,order));assert all(torch.equal(original[k],cached[k]) for k in KEYS)
 with torch.no_grad():
  p=model(original["input"],original["q"],original["mask"],original["bands"]);q=model(cached["input"],cached["q"],cached["mask"],cached["bands"])
 assert torch.equal(p,q)
 offload_cache(a);restored=next(batches(a,order));assert all(torch.equal(original[k],restored[k]) for k in KEYS)
 write(OUT/"resource_gates.json",dict(passed=True,prefix_cache=info,mixed_batch_tensors_bitwise_identical=True,predictions_bitwise_identical=True,offload_values_bitwise_identical=True,cuda_rng_unchanged=True))
 print(info,flush=True)
if __name__=="__main__":main()
