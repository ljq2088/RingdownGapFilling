"""Detached, resumable equal-budget raw/white DGF transfer experiment.
Test targets are read once, after BOTH variants finish selecting checkpoints.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('NUMBA_NUM_THREADS','1');os.environ.setdefault('MPLBACKEND','Agg')
import time,json,traceback,copy,zipfile,signal,subprocess,fcntl
from pathlib import Path
import numpy as np
import torch
import h5py
from .common import N,FS,SEED,sha,write,Preprocess,metric,summary
from .prepare import OUT,PARENT,ROOT,COUNTS,crop_meta,prepare
from .model import PhysicalDGF
from .resources import GPU_MEMORY_FRACTION,cache_on_gpu,guard_headroom,slice_to_gpu
BASE=ROOT/'saved_models/noise/lowSNR/model_with_noise_gaps_HugeRangeParas_14.pth'
LEGACY_REFINER=ROOT/'FillingGap/results_augmented_longrun_20261008/best_refiner.pth'
START=time.time();BATCH=128;MAX_EPOCHS=600;MIN_EPOCHS=100;PATIENCE=100;DELTA=1e-5

def status(stage,**kw):
 v=dict(stage=stage,pid=os.getpid(),updated_unix=time.time(),elapsed_seconds=time.time()-START,**kw)
 write(OUT/os.environ.get('PRD_STATUS_FILE','status.json'),v);print(json.dumps(v),flush=True)

def save(path,v):
 tmp=path.with_suffix('.tmp');torch.save(v,tmp);tmp.replace(path)

def load(split,kind):
 with h5py.File(OUT/(split+'_'+kind+'_features.h5'),'r') as f:
  a={k:torch.from_numpy(f[k][:]) for k in ['input','target','q']}
 with h5py.File(OUT/(split+'_crops.h5'),'r') as f:
  a['mask']=torch.from_numpy(f['mask'][:]);a['physical_truth']=f['truth'][:].astype(float)
 return a

def batches(a,indices):
 for lo in range(0,len(indices),BATCH):
  ix=indices[lo:lo+BATCH]
  yield slice_to_gpu(a,ix)

def lossfn(p,h,m,w,time_transfer=None):
 P,H=torch.fft.rfft(p,dim=-1),torch.fft.rfft(h,dim=-1)
 tt=(H.abs().square()*w).sum((-2,-1));pp=(P.abs().square()*w).sum((-2,-1))
 cross=((H*P.conj()).real*w).sum((-2,-1))
 overlap=1-cross/(tt*pp).clamp_min(1e-24).sqrt()
 spectral=((P-H).abs().square()*w).sum((-2,-1))/tt.clamp_min(1e-12)
 if time_transfer is not None:
  p=torch.fft.irfft(P*time_transfer,n=N,dim=-1);h=torch.fft.irfft(H*time_transfer,n=N,dim=-1)
 error=(p-h).square();den=h.square().sum((-2,-1)).clamp_min(1e-12)
 relative=(error.sum((-2,-1))+2*(error*m[:,None,:]).sum((-2,-1)))/den
 # Both variants use the same physical loss. Tiny numeric guards affect training only.
 return (overlap+.5*spectral+.5*relative).mean()

def predict(model,a,pre):
 model.eval();arr=[]
 with torch.no_grad():
  for v in batches(a,np.arange(len(a['input']))):
   p=model(v['input'],v['q'],v['mask']).cpu().numpy()
   if not np.isfinite(p).all():raise ValueError('nonfinite prediction')
   arr.append(pre.decode(p))
 return np.concatenate(arr)

def evaluate(model,a,pre,psd):
 p=predict(model,a,pre);h=a['physical_truth'];mm=metric(h,p,psd)
 return dict(score=float(mm.mean()),mean_time_mismatch=float(metric(h,p).mean()),median_psd=float(np.median(mm)),p95_psd=float(np.quantile(mm,.95)),relative_l2_mean=float(np.mean(np.linalg.norm((p-h).reshape(len(h),-1),axis=1)/np.linalg.norm(h.reshape(len(h),-1),axis=1))))

def legacy_gate():
 from model.mask_aware_refiner import RefinedDGF
 from model.QTranTimeMixerMod import QTransformModule
 src=Path('/home/ljq/code/RingdownGapFilling/FillingGap/results_augmented_longrun_20261008')
 with np.load(src/'final_test_baseline.npz') as f:x=f['observation'][:8];mask=f['mask'][:8]
 with np.load(src/'final_test_predictions.npz') as f:expected=f['augmented'][:8]
 model=RefinedDGF.from_checkpoints(BASE,LEGACY_REFINER,'cuda');q=QTransformModule()
 with torch.no_grad():
  u=torch.from_numpy(x).cuda();qt=torch.stack([q(torch.from_numpy(row)) for row in x]).cuda()
  p=model(u,qt,torch.from_numpy(mask).cuda()).cpu().numpy()
 err=float(np.linalg.norm(p.astype(float)-expected)/np.linalg.norm(expected))
 if err>1e-5:raise ValueError(('restored legacy dependencies fail frozen prediction reproduction',err))
 write(OUT/'legacy_inference_gate.json',dict(passed=True,n=8,relative_l2=err,base_sha256=sha(BASE),refiner_sha256=sha(LEGACY_REFINER)))
 del model;torch.cuda.empty_cache()

def _train(kind):
 path=OUT/kind;path.mkdir(exist_ok=True)
 if (path/'selected.json').exists():return json.loads((path/'selected.json').read_text())
 torch.manual_seed(SEED);np.random.seed(SEED)
 model=PhysicalDGF(BASE).cuda();psd=np.load(OUT/'calibration_psd.npz')['psd'];pre=Preprocess(psd,kind)
 w=torch.tensor(pre.weights(),device='cuda')
 transfer=pre.sigma[:,None] if kind=='raw' else pre.asd
 time_transfer=torch.tensor(transfer/pre.sigma.mean(),device='cuda',dtype=torch.float32)
 tr=load('train',kind);va=load('validation',kind)
 residency=cache_on_gpu([tr,va]);write(path/'resource_status.json',residency)
 status('resource_configuration',variant=kind,**residency)
 opt=torch.optim.AdamW([{'params':model.base.parameters(),'lr':2e-5},{'params':model.refiner.parameters(),'lr':1e-4}],weight_decay=1e-4)
 sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=20,threshold=DELTA,threshold_mode='abs',min_lr=[2e-7,1e-6])
 best=float('inf');best_epoch=0;stale=0;history=[];first=1
 last=path/'last_checkpoint.pth'
 if last.exists():
  ck=torch.load(last,map_location='cuda',weights_only=False);model.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer']);sch.load_state_dict(ck['scheduler'])
  best=ck['best'];best_epoch=ck['best_epoch'];stale=ck['stale'];first=ck['epoch']+1
  torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state(ck['cuda_rng'].cpu());np.random.set_state(ck['numpy_rng'])
  history=json.loads((path/'history.json').read_text())
 else:
  initial=evaluate(model,va,pre,psd);write(path/'initial_validation.json',initial);best=initial['score']
  save(path/'best.pth',dict(state_dict=model.state_dict(),epoch=0,score=best,kind=kind))
 # No stopping before min epochs; test is not loaded in this function.
 for epoch in range(first,MAX_EPOCHS+1):
  offload=guard_headroom([tr,va])
  if offload is not None:
   residency=offload;write(path/'resource_status.json',residency);status('resource_fallback',variant=kind,**residency)
  t=time.time();model.train();order=torch.randperm(len(tr['input'])).numpy();total=0.;done=0;last_report=0
  for v in batches(tr,order):
   opt.zero_grad(set_to_none=True);p=model(v['input'],v['q'],v['mask']);loss=lossfn(p,v['target'],v['mask'],w,time_transfer)
   if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
   done+=len(v['input']);total+=float(loss.detach())*len(v['input'])
   if time.time()-last_report>45:
    torch.cuda.synchronize();elapsed=time.time()-t;rate=elapsed/max(done,1)
    status('training',variant=kind,epoch=epoch,sources_done=done,sources_total=len(order),epoch_remaining_seconds=(len(order)-done)*rate,best_score=best,best_epoch=best_epoch,stale_epochs=stale,gpu_peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20)
    last_report=time.time()
  result=evaluate(model,va,pre,psd);score=result['score'];sch.step(score)
  if score<best-DELTA:
   best=score;best_epoch=epoch;stale=0
   save(path/'best.pth',dict(state_dict=model.state_dict(),epoch=epoch,score=score,kind=kind))
  else:stale+=1
  secs=time.time()-t;row=dict(epoch=epoch,train_loss=total/len(order),**result,best_epoch=best_epoch,best_score=best,stale_epochs=stale,epoch_seconds=secs,resource_mode=residency['mode'],gpu_allocated_mb=torch.cuda.memory_allocated()/2**20,gpu_reserved_mb=torch.cuda.memory_reserved()/2**20,lr=[g['lr'] for g in opt.param_groups]);history.append(row);write(path/'history.json',history)
  save(last,dict(state_dict=model.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),epoch=epoch,best=best,best_epoch=best_epoch,stale=stale,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),numpy_rng=np.random.get_state()))
  median=float(np.median([x['epoch_seconds'] for x in history[-10:]]))
  status('epoch_complete',variant=kind,**row,median_epoch_seconds=median,minimum_remaining_seconds=max(MIN_EPOCHS-epoch,PATIENCE-stale,0)*median,max_remaining_seconds=(MAX_EPOCHS-epoch)*median)
  if (epoch>=MIN_EPOCHS and stale>=PATIENCE):break
  if (OUT/'STOP').exists():
   status('paused_by_STOP',variant=kind,epoch=epoch);raise SystemExit(0)
 result=dict(variant=kind,selected_epoch=best_epoch,finished_epoch=epoch,best_validation_score=best,stop_reason='patience' if stale>=PATIENCE else 'max_epochs',checkpoint_sha256=sha(path/'best.pth'))
 write(path/'selected.json',result);del model,opt,sch,tr,va;torch.cuda.empty_cache();return result

def train(kind):
 # One writer per variant even when two workers run concurrently.
 path=OUT/kind;path.mkdir(exist_ok=True)
 with open(path/'training.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  return _train(kind)

def start_white_worker():
 path=OUT/'white';path.mkdir(exist_ok=True)
 if (path/'selected.json').exists():return
 meta=OUT/'parallel_worker.json'
 if meta.exists():
  old=json.loads(meta.read_text());proc=Path('/proc')/str(old['pid'])/'cmdline'
  if proc.exists() and 'prd_physical.train_variant' in proc.read_bytes().decode(errors='replace'):return
 env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['PRD_STATUS_FILE']='white/status.json'
 with open(path/'pipeline.log','ab',buffering=0) as log:
  child=subprocess.Popen(['/home/ljq/miniconda3/envs/gw_ml_env/bin/python','-u','-m','prd_physical.train_variant','white'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 write(meta,dict(pid=child.pid,variant='white',started_unix=time.time(),execution='parallel independent training; same batch/seed/loss and separate status/checkpoints'))
 status('parallel_worker_started',white_pid=child.pid)

def final_test(selected):
 # Freeze all choices BEFORE the first read of test truth/noise.
 preferred=min(selected,key=lambda x:x['best_validation_score'])['variant']
 write(OUT/'selection_frozen.json',dict(preferred=preferred,variants=selected,selected_unix=time.time(),test_read=False))
 psd=np.load(OUT/'calibration_psd.npz')['psd'];h=[];noise=[];masks=[];seeds=[];parameters=[]
 with h5py.File(PARENT/'test.h5','r') as f:
  for i in range(COUNTS['test']):
   par=f['parameters'][i];seed=int(f['seeds'][i]);s,m=crop_meta(par,seed)
   h.append(f['truth'][i,:,s:s+N]);noise.append(f['noise'][i,:,s:s+N]);masks.append(m);seeds.append(seed);parameters.append(par)
 h=np.stack(h).astype(float);noise=np.stack(noise);masks=np.stack(masks);x=h*(1-masks[:,None,:])+noise
 from model.QTranTimeMixerMod import QTransformModule
 q=QTransformModule();preds={}
 for kind in ['raw','white']:
  status('final_test',variant=kind);pre=Preprocess(psd,kind);u=pre.encode(x);v=pre.encode(h)
  scales=np.sqrt((u.astype(float)**2).mean(-1)).clip(1e-8)
  qt=np.stack([torch.stack([q(torch.from_numpy(row/sc)) for row,sc in zip(a,b)]).numpy() for a,b in zip(u,scales)])
  a=dict(input=torch.from_numpy(u),target=torch.from_numpy(v),q=torch.from_numpy(qt),mask=torch.from_numpy(masks),physical_truth=h)
  model=PhysicalDGF(BASE).cuda();ck=torch.load(OUT/kind/'best.pth',map_location='cuda',weights_only=True);model.load_state_dict(ck['state_dict'])
  preds[kind]=predict(model,a,pre);del model,a,qt;torch.cuda.empty_cache()
 # Published frozen checkpoints are compared on exactly the same physical data.
 from model.mask_aware_refiner import RefinedDGF
 model=RefinedDGF.from_checkpoints(BASE,LEGACY_REFINER,'cuda');base=[];ref=[]
 with torch.no_grad():
  for lo in range(0,len(h),BATCH):
   a=x[lo:lo+BATCH].astype(np.float32);flat=a.reshape(-1,N)
   qt=torch.stack([q(torch.from_numpy(row)) for row in flat]).cuda();u=torch.from_numpy(flat).cuda();mask=torch.from_numpy(np.repeat(masks[lo:lo+BATCH],2,axis=0)).cuda()
   b=model.baseline(u,qt);p=model.refiner(u*1e22,b*1e22,mask)/1e22
   base.append(b.cpu().numpy().reshape(-1,2,N));ref.append(p.cpu().numpy().reshape(-1,2,N))
 preds['legacy_DGF']=np.concatenate(base);preds['legacy_augmented']=np.concatenate(ref)
 preds['observation']=x
 interp=x.copy();grid=np.arange(N)
 for i in range(len(h)):
  m=masks[i].astype(bool)
  for c in range(2):interp[i,c,m]=np.interp(grid[m],grid[~m],x[i,c,~m])
 preds['linear_interpolation']=interp
 per={};report={};hann=np.hanning(N);hpsd=np.load(OUT/'calibration_psd.npz')['hann_psd']
 peaks=np.argmax((h*h).sum(1),axis=-1);pg=masks[np.arange(len(h)),peaks].astype(bool)
 for name,p in preds.items():
  mm=metric(h,p,psd);tm=metric(h,p);hm=metric(h*hann,p*hann,hpsd)
  rel=np.linalg.norm((p-h).reshape(len(h),-1),axis=1)/np.linalg.norm(h.reshape(len(h),-1),axis=1)
  per[name+'__psd']=mm;per[name+'__time']=tm;per[name+'__hann_psd']=hm;per[name+'__relative_l2']=rel
  report[name]=dict(psd=summary(mm),time=summary(tm),hann_psd=summary(hm),relative_l2=summary(rel),peak_in_gap_psd=summary(mm[pg]))
 paired={};rng=np.random.default_rng(SEED)
 for ref in ['legacy_DGF','legacy_augmented','raw' if preferred=='white' else 'white','linear_interpolation']:
  d=per[preferred+'__psd']-per[ref+'__psd'];bs=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(5000)])
  paired[ref]=dict(mean_delta=float(d.mean()),ci95=np.quantile(bs,[.025,.975]).tolist())
 result=dict(preferred_from_validation=preferred,selected=selected,n=len(h),all_rows_retained=True,observation='G*h+n',time_phase_maximized=False,results=report,paired=paired,improved_over_published_on_same_physical_test=paired['legacy_augmented']['ci95'][1]<0,scope='528-s physical TDI segments; synthetic amplitude/phase prescription; sample-bootstrap conditional on fitted checkpoints; not training-seed uncertainty',historical_legacy_mean=.005368768966470572,cross_distribution_comparison='historical legacy mean is contextual only; not a same-condition improvement claim')
 np.savez_compressed(OUT/'per_example_test.npz',seeds=np.array(seeds),parameters=np.array(parameters),peak_in_gap=pg,**per)
 np.savez_compressed(OUT/'test_predictions.npz',truth=h,noise=noise,mask=masks,**preds)
 write(OUT/'final_results.json',result);status('completed',preferred=preferred,n=len(h),improved_same_test=result['improved_over_published_on_same_physical_test'])

def main():
 OUT.mkdir(exist_ok=True);torch.set_num_threads(2)
 assert torch.cuda.is_available()
 # Use free VRAM for cached data while retaining explicit allocator/headroom caps.
 torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION,0)
 files=[*list((ROOT/'prd_physical').glob('*.py')),*list((ROOT/'model').rglob('*.py')),*list((ROOT/'utils').glob('*.py')),ROOT/'config/config.py']
 manifest={str(p.relative_to(ROOT)):sha(p) for p in files}
 if (OUT/'runtime_source_manifest.json').exists():
  if json.loads((OUT/'runtime_source_manifest.json').read_text())!=manifest:raise RuntimeError('runtime sources changed; use a new output directory')
 else:
  write(OUT/'runtime_source_manifest.json',manifest)
  with zipfile.ZipFile(OUT/'runtime_sources.zip','w',zipfile.ZIP_DEFLATED) as z:
   for p in files:z.write(p,str(p.relative_to(ROOT)))
 status('preparing',gpu=torch.cuda.get_device_name(),gpu_memory_fraction=GPU_MEMORY_FRACTION)
 legacy_gate();prepare()
 for name in ['physics_gates.json','metric_gates.json','training_probe.json','prepared.json']:
  if not json.loads((OUT/name).read_text()).get('passed'):raise RuntimeError('pre-training gate failed: '+name)
 write(OUT/'training_config.json',dict(max_epochs=MAX_EPOCHS,min_epochs=MIN_EPOCHS,patience=PATIENCE,min_delta=DELTA,batch_sources=BATCH,train_unique=COUNTS['train'],validation_unique=COUNTS['validation'],test_unique=COUNTS['test'],observation='G*h+n',variants=['raw','white'],learning_rates=[2e-5,1e-4],schedule='ReduceLROnPlateau factor0.5 patience20',seed=SEED,base_sha256=sha(BASE),legacy_refiner_sha256=sha(LEGACY_REFINER),train_loss='joint direct PSD mismatch + 0.5 PSD relative squared error + 0.5 time/gap relative squared error; same physical-space loss for raw/white variants; training numeric norm guards 1e-12; no evaluation ratio floor',model='unchanged original double-branch DGF + 25489-parameter mask adapter; shared A/E weights, channels combined only in loss/metric',selection='lower all-population joint A/E validation direct PSD mismatch; equal max/min/patience budget'))
 start_white_worker()
 selected=[train(kind) for kind in ['raw','white']]
 final_test(selected)

if __name__=='__main__':
 try:main()
 except BaseException as e:
  if not isinstance(e,SystemExit):status('failed',error=traceback.format_exc())
  raise
