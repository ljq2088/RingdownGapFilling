"""Verify fixed db4 cache equivalence, gradients, memory and throughput."""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import torch,numpy as np,time,statistics
from .run import load,BASE,OUT,lossfn
from .model import PhysicalDGF
from .band_cache import attach_bands
from .resources import GPU_MEMORY_FRACTION,cache_on_gpu,slice_to_gpu,data_keys
from .common import Preprocess,write,sha

def main():
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION);results={}
 for kind in ['raw','white']:
  cp=OUT/kind/'last_checkpoint.pth';before=sha(cp);tr=load('train',kind);va=load('validation',kind)
  model=PhysicalDGF(BASE).cuda();ck=torch.load(cp,map_location='cpu',weights_only=False);model.load_state_dict(ck['state_dict'])
  start=time.monotonic();info={split:attach_bands(ds,kind,split,model) for split,ds in [('train',tr),('validation',va)]};prep_seconds=time.monotonic()-start
  residency=cache_on_gpu([tr,va]);assert residency['mode']=='gpu',residency
  psd=np.load(OUT/'calibration_psd.npz')['psd'];pre=Preprocess(psd,kind);w=torch.tensor(pre.weights(),device='cuda');transfer=pre.sigma[:,None] if kind=='raw' else pre.asd;transfer=torch.tensor(transfer/pre.sigma.mean(),device='cuda',dtype=torch.float32)
  cases=[];torch.cuda.reset_peak_memory_stats()
  for count in [128,96]:
   ix=np.random.default_rng(555).permutation(len(tr['input']))[:count];v=slice_to_gpu(tr,ix);model.eval()
   with torch.no_grad():
    fresh=model(v['input'],v['q'],v['mask']);cached=model(v['input'],v['q'],v['mask'],v['bands'])
   relative=float(torch.linalg.vector_norm(cached-fresh)/torch.linalg.vector_norm(fresh));assert relative<1e-6,relative
   model.train();cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone();model.zero_grad(set_to_none=True)
   l0=lossfn(model(v['input'],v['q'],v['mask']),v['target'],v['mask'],w,transfer);l0.backward();grads={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
   model.zero_grad(set_to_none=True);torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
   l1=lossfn(model(v['input'],v['q'],v['mask'],v['bands']),v['target'],v['mask'],w,transfer);l1.backward()
   numerator=denominator=0.
   for n,p in model.named_parameters():
    assert (p.grad is not None)==(n in grads)
    if p.grad is not None:
     numerator+=float((p.grad.double()-grads[n].double()).square().sum());denominator+=float(grads[n].double().square().sum())
   grad_relative=(numerator/denominator)**.5;assert grad_relative<1e-5,grad_relative
   assert abs(float(l1.detach())-float(l0.detach()))<1e-5
   cases.append(dict(batch_sources=count,inference_bitwise_equal=bool(torch.equal(fresh,cached)),inference_relative_l2=relative,gradient_relative_l2=grad_relative,loss_absolute_difference=abs(float(l1.detach())-float(l0.detach()))))
   del grads,fresh,cached,l0,l1
  # Identical batch128, cloned model, no saved optimizer/checkpoint updated.
  v=slice_to_gpu(tr,np.arange(128));timings={}
  for use_cache in [False,True]:
   times=[]
   for step in range(8):
    model.zero_grad(set_to_none=True);torch.cuda.synchronize();t=time.monotonic()
    pred=model(v['input'],v['q'],v['mask'],v['bands'] if use_cache else None);loss=lossfn(pred,v['target'],v['mask'],w,transfer);loss.backward();torch.cuda.synchronize();times.append(time.monotonic()-t)
   timings['cached' if use_cache else 'fresh']=statistics.median(times[2:])
  assert sha(cp)==before
  results[kind]=dict(cases=cases,cache_generation_seconds=prep_seconds,cache=info,residency=residency,forward_backward_seconds=timings,speedup=timings['fresh']/timings['cached'],peak_gpu_mb=torch.cuda.max_memory_allocated()/2**20,checkpoint_sha256=before,checkpoint_unmodified=True)
  print(kind,results[kind],flush=True)
  del tr,va,model,ck,v,pred,loss,w,transfer;torch.cuda.empty_cache()
 write(OUT/'band_cache_gates.json',dict(passed=True,results=results,scope='execution-equivalence check; no test targets read and no model training result asserted'))

if __name__=='__main__':main()
