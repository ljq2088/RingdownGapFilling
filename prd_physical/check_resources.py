"""Verify cache-only execution change without advancing any saved model."""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import torch,numpy as np
from . import resources
from .run import load,BASE,OUT,lossfn
from .model import PhysicalDGF
from .common import Preprocess,write,sha

def main():
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(resources.GPU_MEMORY_FRACTION)
 path=OUT/'raw/last_checkpoint.pth';before=sha(path)
 tr,va=load('train','raw'),load('validation','raw')
 ix=np.random.default_rng(901).permutation(len(tr['input']))[:128]
 a=resources.slice_to_gpu(tr,ix)
 model=PhysicalDGF(BASE).cuda();ck=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(ck['state_dict']);model.eval()
 with torch.no_grad():p0=model(a['input'],a['q'],a['mask']).clone()
 torch.cuda.reset_peak_memory_stats();info=resources.cache_on_gpu([tr,va]);assert info['mode']=='gpu',info
 b=resources.slice_to_gpu(tr,ix)
 assert all(torch.equal(a[k],b[k]) for k in resources.KEYS)
 with torch.no_grad():p1=model(b['input'],b['q'],b['mask'])
 assert torch.equal(p0,p1),'cache changed inference output'
 psd=np.load(OUT/'calibration_psd.npz')['psd'];pre=Preprocess(psd,'raw');w=torch.tensor(pre.weights(),device='cuda');transfer=torch.tensor(pre.sigma[:,None]/pre.sigma.mean(),device='cuda',dtype=torch.float32)
 opt=torch.optim.AdamW(model.parameters(),lr=2e-5);model.train();opt.zero_grad(set_to_none=True);loss=lossfn(model(b['input'],b['q'],b['mask']),b['target'],b['mask'],w,transfer)
 assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();torch.cuda.synchronize()
 peak=torch.cuda.max_memory_allocated()/2**20
 # Exercise the real fallback path by temporarily raising its threshold,
 # without exhausting memory or touching any other process.
 threshold=resources.MIN_FREE_BYTES;resources.MIN_FREE_BYTES=torch.cuda.mem_get_info()[1]
 fallback=resources.guard_headroom([tr,va]);resources.MIN_FREE_BYTES=threshold
 assert fallback and all(not ds['input'].is_cuda for ds in [tr,va])
 c=resources.slice_to_gpu(tr,ix);assert all(torch.equal(a[k],c[k]) for k in resources.KEYS)
 assert sha(path)==before,'saved checkpoint changed'
 report=dict(passed=True,cache=info,batch_sources=128,input_bitwise_equal=True,inference_bitwise_equal=True,CUDA_RNG_unchanged=info['rng_unchanged'],finite_training_loss=True,training_peak_gpu_mb=peak,forced_threshold_offload_passed=True,checkpoint_sha256=before,checkpoint_unmodified=True,guard_free_GiB=threshold/2**30,allocator_fraction=resources.GPU_MEMORY_FRACTION)
 write(OUT/'resource_change_gates.json',report);print(report,flush=True)

if __name__=='__main__':main()
