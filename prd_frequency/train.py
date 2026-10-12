"""Resumable training workers; no test-target access in training or checkpoint selection."""
import os
os.environ.setdefault("OMP_NUM_THREADS","2");os.environ.setdefault("NUMBA_NUM_THREADS","1")
import sys,time,json,traceback,copy,fcntl,subprocess,psutil
import numpy as np,torch
from prd_physical.common import metric,summary,Preprocess,write,sha,N
from prd_physical.resources import available_gpu_bytes
from .settings import *
from .model import FrequencyDGF
from prd_optimize.loss import Objective
from .data import load_maps,batches,physical_truth,predict,cache_prefix,offload_cache

def status(path,stage,**kw):
 v=dict(stage=stage,pid=os.getpid(),updated_unix=time.time(),**kw);write(path/"status.json",v);print(json.dumps(v),flush=True)
def save(path,v):
 tmp=path.with_suffix(".tmp");torch.save(v,tmp);tmp.replace(path)
def validate(model,a,truth,pre,cal):
 p=predict(model,a,pre);mm=metric(truth,p,cal["psd"]);hann=np.hanning(N)
 H=np.fft.rfft(truth,axis=-1)
 snr=np.sqrt((4/(2*N)*(abs(H[...,1:-1])**2/cal["psd"][...,1:-1])).sum((1,2)))
 low=snr<20
 return dict(score=float(mm.mean()),hann_mean=float(metric(truth*hann,p*hann,cal["hann_psd"]).mean()),time_mean=float(metric(truth,p).mean()),median=float(np.median(mm)),p95=float(np.quantile(mm,.95)),snr_lt20_n=int(low.sum()),snr_lt20_mean=float(mm[low].mean()),snr_ge20_mean=float(mm[~low].mean()))
def optimizer(model):
 groups=[dict(params=model.backbone.dgf.base.parameters(),lr=5e-7),dict(params=model.backbone.dgf.refiner.parameters(),lr=2.5e-6),dict(params=model.backbone.joint.parameters(),lr=2.5e-5)]
 if model.frequency is not None:groups.append(dict(params=model.frequency.parameters(),lr=1e-4))
 return torch.optim.AdamW(groups,weight_decay=1e-4)
def check_sources():
 manifest=json.loads((OUT/"runtime_source_manifest.json").read_text())
 for name,expected in manifest.items():
  if sha(ROOT/name)!=expected:raise RuntimeError("source changed; cannot resume "+name)
 for f in ["prepared.json","preflight.json","cache_gates.json","resource_gates.json"]:
  if not json.loads((OUT/f).read_text()).get("passed"):raise RuntimeError("preflight failure "+f)
def resource_wait(path,epoch,datasets):
 if available_gpu_bytes()[0]<MIN_GPU_FREE:
  for a in datasets:offload_cache(a)
  torch.cuda.empty_cache()
 start=time.monotonic()
 while available_gpu_bytes()[0]<MIN_GPU_FREE or psutil.virtual_memory().available<MIN_RAM_FREE:
  torch.cuda.empty_cache()
  if available_gpu_bytes()[0]>=MIN_GPU_FREE and psutil.virtual_memory().available>=MIN_RAM_FREE:return
  status(path,"waiting_for_gpu_headroom",epoch=epoch,minimum_free_gib=MIN_GPU_FREE/2**30,minimum_ram_gib=MIN_RAM_FREE/2**30)
  if (OUT/"STOP").exists():raise SystemExit(0)
  time.sleep(15)
  if time.monotonic()-start>600:raise RuntimeError("RAM/VRAM remains below declared safe headroom for ten minutes")
def main(kind):
 if kind not in VARIANTS:raise ValueError(kind)
 path=OUT/kind;path.mkdir(exist_ok=True)
 with open(path/"training.lock","a") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (path/"selected.json").exists():return
  check_sources();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(GPU_FRACTION)
  torch.manual_seed(SEED);np.random.seed(SEED)
  model=FrequencyDGF(kind).cuda();ema=copy.deepcopy(model).eval()
  for param in ema.parameters():param.requires_grad_(False)
  cal=np.load(OLD/"calibration_psd.npz");pre=Preprocess(cal["psd"],"white");objective=Objective(pre,cal,"joint_hann")
  tr=load_maps("train");va=load_maps("validation");truth=physical_truth("validation")
  cache_info={"validation":cache_prefix(va,VAL_N),"train":cache_prefix(tr,4096)};write(path/"resource_configuration.json",cache_info)
  opt=optimizer(model);sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",factor=.5,patience=15,threshold=DELTA,threshold_mode="abs",min_lr=[1e-8,1e-7,1e-6]+([5e-6] if model.frequency is not None else []))
  torch.manual_seed(SEED+1);order_rng=torch.Generator().manual_seed(SEED+2)
  best=float("inf");best_epoch=0;best_type="online";stale=0;history=[];first=1;steps=0
  last=path/"last_checkpoint.pth"
  if last.exists():
   c=torch.load(last,map_location="cuda",weights_only=False)
   model.load_state_dict(c["state_dict"]);ema.load_state_dict(c["ema"]);opt.load_state_dict(c["optimizer"]);sch.load_state_dict(c["scheduler"])
   best=c["best"];best_epoch=c["best_epoch"];best_type=c["best_type"];stale=c["stale"];first=c["epoch"]+1;steps=c["steps"]
   torch.set_rng_state(c["torch_rng"].cpu());torch.cuda.set_rng_state(c["cuda_rng"].cpu());order_rng.set_state(c["order_rng"].cpu());np.random.set_state(c["numpy_rng"])
   history=json.loads((path/"history.json").read_text());status(path,"resumed",epoch=first-1,best_score=best)
  else:
   initial=validate(model,va,truth,pre,cal);best=initial["score"];write(path/"initial_validation.json",initial)
   save(path/"best.pth",dict(state_dict=model.state_dict(),epoch=0,score=best,kind=kind,weight_type="online"))
  for epoch in range(first,MAX_EPOCHS+1):
   if (OUT/"STOP").exists():status(path,"paused_by_STOP",epoch=epoch-1);return
   resource_wait(path,epoch,[tr,va]);t=time.monotonic();model.train();order=torch.randperm(TRAIN_N,generator=order_rng).numpy();total=0.;done=0;last_report=0
   for v in batches(tr,order):
    opt.zero_grad(set_to_none=True);prediction=model(v["input"],v["q"],v["mask"],v["bands"],v["q_low"]);loss=objective(prediction,v["target"],v["mask"])
    if not torch.isfinite(loss):raise ValueError("nonfinite loss")
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();steps+=1
    with torch.no_grad():
     for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,.005)
     for a,b in zip(ema.buffers(),model.buffers()):a.copy_(b)
    count=len(v["input"]);done+=count;total+=float(loss.detach())*count
    if time.monotonic()-last_report>40:
     elapsed=time.monotonic()-t;status(path,"training",variant=kind,epoch=epoch,sources_done=done,sources_total=TRAIN_N,epoch_remaining_seconds=(TRAIN_N-done)*elapsed/max(done,1),best_score=best,best_epoch=best_epoch,stale_epochs=stale,gpu_peak_mb=torch.cuda.max_memory_allocated()/2**20);last_report=time.monotonic()
   online=validate(model,va,truth,pre,cal);averaged=validate(ema,va,truth,pre,cal)
   typ,current=("ema",averaged) if averaged["score"]<online["score"] else ("online",online)
   score=current["score"];sch.step(score)
   if score<best-DELTA:
    best=score;best_epoch=epoch;best_type=typ;stale=0;chosen=ema if typ=="ema" else model
    save(path/"best.pth",dict(state_dict=chosen.state_dict(),epoch=epoch,score=best,kind=kind,weight_type=typ))
   else:stale+=1
   secs=time.monotonic()-t
   row=dict(epoch=epoch,train_loss=total/TRAIN_N,score=score,online=online,ema=averaged,current_weight_type=typ,best_score=best,best_epoch=best_epoch,best_type=best_type,stale_epochs=stale,epoch_seconds=secs,lr=[g["lr"] for g in opt.param_groups],gpu_peak_mb=torch.cuda.max_memory_allocated()/2**20)
   history.append(row);write(path/"history.json",history)
   save(last,dict(state_dict=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),epoch=epoch,steps=steps,best=best,best_epoch=best_epoch,best_type=best_type,stale=stale,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),numpy_rng=np.random.get_state(),order_rng=order_rng.get_state()))
   median=float(np.median([v["epoch_seconds"] for v in history[-10:]]))
   status(path,"epoch_complete",variant=kind,**row,median_epoch_seconds=median,max_remaining_seconds=(MAX_EPOCHS-epoch)*median,minimum_remaining_seconds=max(MIN_EPOCHS-epoch,PATIENCE-stale,0)*median)
   if epoch>=MIN_EPOCHS and stale>=PATIENCE:break
  result=dict(variant=kind,selected_epoch=best_epoch,finished_epoch=epoch,best_validation_score=best,weight_type=best_type,checkpoint_sha256=sha(path/"best.pth"),stop_reason="patience" if stale>=PATIENCE else "max_epochs")
  write(path/"selected.json",result);status(path,"variant_completed",selection=result)
if __name__=="__main__":
 kind=sys.argv[1];path=OUT/kind;path.mkdir(exist_ok=True)
 try:main(kind)
 except BaseException as e:
  if not isinstance(e,SystemExit):status(path,"failed",error=traceback.format_exc())
  raise
