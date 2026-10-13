"""Small-batch read-only frozen-model predictions; existing workers keep their sources."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('NUMBA_NUM_THREADS','1')
import json,time,shutil,subprocess
import numpy as np,torch
from prd_physical.common import write,sha
from prd_physical.resources import available_gpu_bytes
from prd_frequency.model import FrequencyDGF
from .settings import *
def status(stage,**kw):
 v=dict(stage=stage,pid=os.getpid(),updated_unix=time.time(),**kw);write(OUT/'prediction_status.json',v);print(json.dumps(v),flush=True)
def main():
 OUT.mkdir(exist_ok=True);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(GPU_FRACTION)
 for name,expected in json.loads((FREQUENCY/'runtime_source_manifest.json').read_text()).items():assert sha(ROOT/name)==expected
 path=OUT/'frequency_snapshot.pth'
 if not path.exists():shutil.copy2(FREQUENCY/'frequency_adapter/best.pth',path)
 snapshot=torch.load(path,map_location='cpu',weights_only=True)
 write(OUT/'anchor_manifest.json',dict(previous_checkpoint_sha256='7ab7fd7bbead48e0ef5408ecceaaaadacd9e00eeecd0bf9a28185912e3edcea5',frequency_snapshot_sha256=sha(path),frequency_snapshot_epoch=snapshot['epoch'],frequency_snapshot_validation_score=snapshot['score'],training_indices=[0,PRED_TRAIN_N-1],validation_count=VAL_N,test_read=False))
 t=time.monotonic()
 for variant in ['previous_joint','frequency_snapshot']:
  model=FrequencyDGF('continuation' if variant=='previous_joint' else 'frequency_adapter').cuda().eval()
  if variant=='frequency_snapshot':model.load_state_dict(snapshot['state_dict'])
  for split,count in [('train',PRED_TRAIN_N),('validation',VAL_N)]:
   out=OUT/(variant+'_'+split+'_prediction.npy')
   if out.exists():assert np.load(out,mmap_mode='r').shape==(count,2,1056);continue
   maps={k:np.load(CACHE/(split+'_'+k+'.npy'),mmap_mode='c') for k in ['input','q','q_low','mask','bands']}
   tmp=out.with_suffix('.tmp.npy');a=np.lib.format.open_memmap(tmp,mode='w+',dtype='f4',shape=(count,2,1056));last=0
   with torch.inference_mode():
    for lo in range(0,count,BATCH):
     # Opportunistic inference preserves the original global 4-GiB free guard.
     free,_=available_gpu_bytes()
     if free<4.25*2**30:
      torch.cuda.empty_cache();status('waiting_for_vram',free_gib=free/2**30)
      start=time.monotonic()
      while available_gpu_bytes()[0]<4.25*2**30:
       if time.monotonic()-start>600:raise RuntimeError('insufficient headroom for strategy inference')
       time.sleep(5)
     hi=min(lo+BATCH,count);v={k:torch.from_numpy(maps[k][lo:hi]).cuda() for k in maps}
     y=model(v['input'],v['q'],v['mask'],v['bands'],v['q_low']).cpu().numpy();assert np.isfinite(y).all();a[lo:hi]=y
     if time.monotonic()-last>30 or hi==count:status('predicting',anchor=variant,split=split,completed=hi,total=count,elapsed_seconds=time.monotonic()-t,peak_mb=torch.cuda.max_memory_allocated()/2**20);last=time.monotonic()
   a.flush();del a;tmp.replace(out)
   del maps
  del model;torch.cuda.empty_cache()
 status('predictions_ready',test_read=False)
if __name__=='__main__':main()
