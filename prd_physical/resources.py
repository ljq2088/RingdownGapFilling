"""Execution-only data residency policy; batch/order/objective are unchanged."""
import torch,subprocess
KEYS=('input','target','q','mask')
GPU_MEMORY_FRACTION=.40
MIN_FREE_BYTES=4*2**30
ACTIVATION_RESERVE_BYTES=3*2**30

def available_gpu_bytes():
 cuda_free,total=torch.cuda.mem_get_info()
 try:
  text=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=3).strip().splitlines()[0]
  global_free=int(float(text)*2**20)
 except (OSError,ValueError,subprocess.SubprocessError):
  global_free=cuda_free
 return min(cuda_free,global_free),total

def data_keys(dataset):
 return KEYS+(('bands',) if 'bands' in dataset else ())

def dataset_bytes(datasets):
 return sum(a[k].numel()*a[k].element_size() for a in datasets for k in data_keys(a))

def cpu_cache(datasets):
 for a in datasets:
  for k in data_keys(a):
   if a[k].is_cuda:a[k]=a[k].cpu()
 torch.cuda.empty_cache()

def cache_on_gpu(datasets):
 needed=dataset_bytes(datasets);free,total=available_gpu_bytes()
 if free-needed-ACTIVATION_RESERVE_BYTES<MIN_FREE_BYTES:
  return dict(mode='cpu',cache_bytes=needed,reason='preserve free VRAM plus activation reserve',free_gpu_bytes=free)
 rng=torch.cuda.get_rng_state().clone()
 try:
  for a in datasets:
   for k in data_keys(a):a[k]=a[k].cuda()
 except torch.cuda.OutOfMemoryError:
  cpu_cache(datasets)
  return dict(mode='cpu',cache_bytes=needed,reason='allocation fallback')
 if not torch.equal(rng,torch.cuda.get_rng_state()):raise RuntimeError('cache transfer changed CUDA RNG')
 return dict(mode='gpu',cache_bytes=needed,free_gpu_bytes=available_gpu_bytes()[0],reason='observed available headroom',rng_unchanged=True)

def guard_headroom(datasets):
 if any(a['input'].is_cuda for a in datasets) and available_gpu_bytes()[0]<MIN_FREE_BYTES:
  cpu_cache(datasets)
  return dict(mode='cpu',reason='VRAM free space fell below 4 GiB; automatic offload',free_gpu_bytes=available_gpu_bytes()[0])
 return None

def slice_to_gpu(a,indices):
 # Same source indices/order and values in both paths; no random operation.
 ix=torch.as_tensor(indices,device=a['input'].device,dtype=torch.long)
 return {k:a[k].index_select(0,ix).cuda() for k in data_keys(a)}
