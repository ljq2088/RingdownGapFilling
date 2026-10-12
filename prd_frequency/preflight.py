"""Reproduction, cache transport and frequency coverage checks before long training."""
import os
os.environ.setdefault('NUMBA_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','2')
import json,time
import numpy as np,torch
from ssqueezepy import cwt
from prd_optimize.model import OptimizedDGF
from prd_physical.common import Preprocess,write,sha,N,FS
from prd_optimize.loss import Objective
from .settings import *
from .features import features,SCALES,FREQUENCIES
from .data import load_maps,batches,cache_prefix,offload_cache,physical_truth
from .model import FrequencyDGF
from .train import validate,optimizer

def main():
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(GPU_FRACTION)
 # Earlier caches and runtime source have immutable provenance.
 for name,expected in json.loads((PREVIOUS/'runtime_source_manifest.json').read_text()).items():
  assert sha(ROOT/name)==expected,name
 manifests=json.loads((PREVIOUS/'cache_manifest.json').read_text())
 for name,expected in manifests.items():assert sha(PREVIOUS/name)==expected,name
 a=load_maps('train');va=load_maps('validation');cal=np.load(OLD/'calibration_psd.npz');pre=Preprocess(cal['psd'],'white')
 indices=np.array([0,4095,4096,11999,12000,23456,40000,59999]);v=next(batches(a,indices));torch.manual_seed(SEED)
 model=FrequencyDGF('frequency_adapter').cuda();original=OptimizedDGF('joint_hann').cuda().eval()
 original.load_state_dict(torch.load(START_CHECKPOINT,map_location='cuda',weights_only=True)['state_dict']);model.eval()
 with torch.no_grad():
  expected=original(v['input'],v['q'],v['mask'],v['bands']);actual=model(v['input'],v['q'],v['mask'],v['bands'],v['q_low'])
 assert torch.equal(expected,actual),'zero-head initialization failed exact warm-start reproduction'
 del original;torch.cuda.empty_cache()
 # Batched CWT equals independently recomputed single-channel CWT on selected rows.
 errors=[]
 for i in indices:
  u=a['input'][i].numpy();scale=np.sqrt(np.mean(u.astype(float)**2,axis=-1)).clip(1e-8);rows=[]
  for row,sc in zip(u,scale):
   z,_=cwt(row.astype(np.float64)/sc,'morlet',scales=SCALES);rows.append(np.stack([abs(z),np.angle(z)])[...,::4])
  diff=float(np.max(abs(np.stack(rows)-a['q_low'][i].numpy())));assert diff==0;errors.append(diff)
 initial=validate(model,va,physical_truth('validation'),pre,cal)
 objective=Objective(pre,cal,'joint_hann');model.train();loss=objective(model(v['input'],v['q'],v['mask'],v['bands'],v['q_low']),v['target'],v['mask']);loss.backward()
 head_grad=float(model.frequency.head.weight.grad.norm());assert np.isfinite(head_grad) and head_grad>0
 del model,actual,expected,v;torch.cuda.empty_cache()
 # Mixed cached/noncached values and predictions retain original exact values.
 model=FrequencyDGF('frequency_adapter').cuda().eval();before=torch.cuda.get_rng_state().clone()
 normal=next(batches(a,indices));info=cache_prefix(a,4096);assert info['rows']==4096
 mixed=next(batches(a,indices));assert all(torch.equal(normal[k],mixed[k]) for k in KEYS)
 with torch.no_grad():
  x=model(normal['input'],normal['q'],normal['mask'],normal['bands'],normal['q_low']);y=model(mixed['input'],mixed['q'],mixed['mask'],mixed['bands'],mixed['q_low'])
 assert torch.equal(x,y);assert torch.equal(before,torch.cuda.get_rng_state())
 offload_cache(a);restored=next(batches(a,indices));assert all(torch.equal(normal[k],restored[k]) for k in KEYS)
 train_seeds=np.load(OUT/'cache/train_seeds.npy',mmap_mode='r');val_seeds=np.load(OUT/'cache/validation_seeds.npy',mmap_mode='r')
 assert len(train_seeds)==TRAIN_N and len(np.unique(train_seeds))==TRAIN_N
 assert not np.isin(train_seeds,val_seeds).any()
 for split,n in [('train',TRAIN_N),('validation',VAL_N)]:
  for key in KEYS:
   z=np.load(OUT/'cache'/(split+'_'+key+'.npy'),mmap_mode='r');assert len(z)==n
   for lo in range(0,n,512):assert np.isfinite(z[lo:lo+512]).all(),key
  assert (np.load(OUT/'cache'/(split+'_mask.npy'),mmap_mode='r').sum(-1)==66).all()
 # A full batch backward pass guards the declared allocator fraction with validation cache resident.
 del normal,mixed,restored,x,y,a,model;torch.cuda.empty_cache()
 tr=load_maps('train');va=load_maps('validation');model=FrequencyDGF('frequency_adapter').cuda();opt=optimizer(model)
 resources={'validation':cache_prefix(va,VAL_N),'train':cache_prefix(tr,4096)}
 torch.cuda.reset_peak_memory_stats();v=next(batches(tr,np.arange(BATCH)));opt.zero_grad(set_to_none=True)
 loss=objective(model(v['input'],v['q'],v['mask'],v['bands'],v['q_low']),v['target'],v['mask']);loss.backward();opt.step();torch.cuda.synchronize()
 peak=torch.cuda.max_memory_allocated();assert peak<torch.cuda.get_device_properties(0).total_memory*GPU_FRACTION
 value=dict(passed=True,initial_validation=initial,warm_start_checkpoint_sha256=START_SHA,zero_adapter_prediction_bitwise_equal=True,frequency_adapter_parameters=sum(x.numel() for x in model.frequency.parameters()),head_gradient_norm=head_grad,test_read=False)
 write(OUT/'preflight.json',value)
 write(OUT/'cache_gates.json',dict(passed=True,previous_cache_hashes_verified=True,batched_vs_single_cwt_max_errors=errors,all_rows_finite=True,all_masks_66=True,seeds_disjoint=True,test_read=False))
 write(OUT/'resource_gates.json',dict(passed=True,cache=resources,mixed_batch_bitwise_equal=True,predictions_bitwise_equal=True,offload_bitwise_equal=True,cuda_rng_unchanged=True,batch_backward_peak_mb=peak/2**20,per_worker_fraction=GPU_FRACTION))
 print(json.dumps(value),flush=True);print('resource peak MB',peak/2**20,flush=True)
if __name__=='__main__':main()
