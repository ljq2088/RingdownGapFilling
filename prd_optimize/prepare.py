"""Expand training to every existing source; keep independent new test sealed."""
import os
os.environ.setdefault("OMP_NUM_THREADS","2");os.environ.setdefault("NUMBA_NUM_THREADS","1")
import time,json
from pathlib import Path
import h5py,numpy as np,torch
from prd_physical.common import N,Preprocess,sha,write
from prd_physical.prepare import crop_meta
from model.QTranTimeMixerMod import QTransformModule
from .settings import OUT,OLD,PARENT,TRAIN_N,VAL_N,KEYS
from .model import OptimizedDGF
SHAPES={"input":(2,N),"target":(2,N),"q":(2,2,16,264),"mask":(N,),"bands":(2,8,N),"truth":(2,N),"parameters":(15,),"seeds":()}
def progress(stage,**kw):
 v=dict(stage=stage,pid=os.getpid(),updated_unix=time.time(),**kw);write(OUT/"prepare_status.json",v);print(json.dumps(v),flush=True)
def split_cache(split,count,model,pre,q):
 cache=OUT/"cache";cache.mkdir(exist_ok=True);final=cache/(split+"_complete.json")
 if final.exists():
  v=json.loads(final.read_text());assert v["count"]==count
  for k in SHAPES:assert np.load(cache/(split+"_"+k+".npy"),mmap_mode="r").shape==(count,*SHAPES[k])
  return v
 maps={k:np.lib.format.open_memmap(cache/(split+"_"+k+".npy"),mode="w+",dtype="i8" if k=="seeds" else "f8" if k=="parameters" else "f4",shape=(count,*shape)) for k,shape in SHAPES.items()}
 prefix=12000 if split=="train" else VAL_N
 # Exact reuse of the previous frozen preprocessing for its prefix.
 with h5py.File(OLD/(split+"_white_features.h5"),"r") as features,h5py.File(OLD/(split+"_crops.h5"),"r") as crops,h5py.File(OLD/(split+"_white_db4_bands.h5"),"r") as bands:
  for lo in range(0,prefix,256):
   hi=min(prefix,lo+256)
   for k in ["input","target","q"]:maps[k][lo:hi]=features[k][lo:hi]
   for k in ["mask","truth","parameters","seeds"]:maps[k][lo:hi]=crops[k][lo:hi]
   maps["bands"][lo:hi]=bands["bands"][lo:hi]
  assert bands.attrs["input_features_sha256"]==sha(OLD/(split+"_white_features.h5"))
 t=time.monotonic();last=0
 with h5py.File(PARENT/(split+".h5"),"r") as parent:
  assert len(parent["seeds"])>=count
  for lo in range(prefix,count,128):
   hi=min(lo+128,count);inputs=[]
   for i in range(lo,hi):
    par=parent["parameters"][i];seed=int(parent["seeds"][i]);s,m=crop_meta(par,seed)
    h=parent["truth"][i,:,s:s+N];n=parent["noise"][i,:,s:s+N];x=h*(1-m)+n
    u=pre.encode(x);v=pre.encode(h);scale=np.sqrt(np.mean(u.astype(float)**2,axis=-1)).clip(1e-8)
    qt=torch.stack([q(torch.from_numpy(row/sc)) for row,sc in zip(u,scale)]).numpy()
    for k,z in [("input",u),("target",v),("q",qt),("mask",m),("truth",h),("parameters",par),("seeds",seed)]:maps[k][i]=z
    inputs.append(u)
   with torch.no_grad():maps["bands"][lo:hi]=model.input_bands(torch.from_numpy(np.stack(inputs)).cuda()).cpu().numpy()
   if time.monotonic()-last>30 or hi==count:
    done=hi-prefix;progress("preparing_cache",split=split,completed=hi,total=count,remaining_seconds=(count-hi)*(time.monotonic()-t)/max(1,done));last=time.monotonic()
   if (OUT/"STOP").exists():raise RuntimeError("STOP requested during preparation")
 for v in maps.values():v.flush()
 seeds=maps["seeds"]
 assert len(np.unique(seeds))==count
 with h5py.File(PARENT/(split+".h5"),"r") as f:assert np.array_equal(seeds,f["seeds"][:count])
 for k in KEYS:
  for lo in range(0,count,512):assert np.isfinite(maps[k][lo:lo+512]).all(),k
 assert (maps["mask"].sum(-1)==66).all()
 v=dict(passed=True,count=count,seed_min=int(seeds.min()),seed_max=int(seeds.max()),cache_bytes=sum(x.nbytes for x in maps.values()),old_prefix_reused_exactly=prefix)
 write(final,v);return v

def main():
 OUT.mkdir(exist_ok=True);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.3)
 if (OUT/"prepared.json").exists():return
 cal=np.load(OLD/"calibration_psd.npz");pre=Preprocess(cal["psd"],"white");q=QTransformModule();model=OptimizedDGF("continuation").cuda().eval()
 info={s:split_cache(s,c,model,pre,q) for s,c in [("validation",VAL_N),("train",TRAIN_N)]}
 assert info["train"]["seed_max"]<info["validation"]["seed_min"]
 write(OUT/"prepared.json",dict(passed=True,splits=info,psd_sha256=sha(OLD/"calibration_psd.npz"),test_targets_read=False))
 progress("prepared",**info)
if __name__=="__main__":main()
