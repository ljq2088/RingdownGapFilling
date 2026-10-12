import os
os.environ.setdefault('NUMBA_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','2')
import time,json
from pathlib import Path
import numpy as np
from ssqueezepy import cwt
from ssqueezepy.experimental import scale_to_freq
from prd_physical.common import write,sha,N,FS
from .settings import *
from .features import features,FREQUENCIES,SCALES
def main():
 OUT.mkdir(exist_ok=True);cache=OUT/'cache';cache.mkdir(exist_ok=True)
 assert sha(START_CHECKPOINT)==START_SHA
 assert json.loads((PREVIOUS/'prepared.json').read_text())['passed']
 info={}
 for split,count in [('validation',VAL_N),('train',TRAIN_N)]:
  for src in (PREVIOUS/'cache').glob(split+'_*.npy'):
   dst=cache/src.name
   if not dst.exists():dst.symlink_to(src)
  final=cache/(split+'_low_complete.json')
  if final.exists():info[split]=json.loads(final.read_text());continue
  u=np.load(cache/(split+'_input.npy'),mmap_mode='r');assert len(u)==count
  target=cache/(split+'_q_low.npy');tmp=target.with_suffix('.tmp.npy')
  a=np.lib.format.open_memmap(tmp,mode='w+',dtype='f4',shape=(count,2,2,16,264))
  t=time.monotonic();last=0
  for lo in range(0,count,32):
   hi=min(lo+32,count);a[lo:hi]=features(u[lo:hi]);assert np.isfinite(a[lo:hi]).all()
   if time.monotonic()-last>30 or hi==count:
    v=dict(stage='observed_frequency_features',split=split,completed=hi,total=count,remaining_seconds=(count-hi)*(time.monotonic()-t)/hi,pid=os.getpid())
    write(OUT/'prepare_status.json',v);print(json.dumps(v),flush=True);last=time.monotonic()
  a.flush();del a;tmp.replace(target);target.chmod(0o444)
  v=dict(passed=True,count=count,sha256=sha(target),feature_source='whitened observed G*h+n / observed channel RMS',target_parameters_used=False)
  write(final,v);info[split]=v
 write(OUT/'prepared.json',dict(passed=True,splits=info,previous_prepared_sha256=sha(PREVIOUS/'prepared.json'),test_targets_read=False,frequencies_hz=FREQUENCIES.tolist(),scales=SCALES.tolist(),kernel_peak_hz=scale_to_freq(SCALES,'morlet',N,fs=FS).tolist()))
 print('prepared',flush=True)
if __name__=='__main__':main()
