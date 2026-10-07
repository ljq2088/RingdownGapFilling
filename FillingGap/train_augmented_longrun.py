"""Continue refinement with train-only noise recombination and new masks.

Twelve thousand independent training waveforms remain twelve thousand unique
waveforms: six combinations per waveform are augmentation, not new sources.
Noise uses the same parameter-independent generator PSD; no test noise is used.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('NUMBA_NUM_THREADS','1')
os.environ.setdefault('MPLBACKEND','Agg')
from pathlib import Path
import sys,json,time,copy,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.signal import hilbert
from model.mask_aware_refiner import MaskAwareRefiner
from FillingGap.compare_mismatch_metrics import mismatch,summary,digest
from FillingGap import train_longrun as previous
OUT=ROOT/'FillingGap/results_augmented_longrun_20261008'
OUT.mkdir(exist_ok=True);previous.OUT=OUT
START=time.time();SEED=2026100831
OLD=ROOT/'FillingGap/results_longrun_20261008'
CP=OLD/'best_refiner.pth';PSD=previous.PSD_PATH
MAX_EPOCHS=1200;MIN_EPOCHS=400;PATIENCE=250;DELTA=1e-5
def write(name,value):
 p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)
def status(stage,**kw):
 d=dict(stage=stage,pid=os.getpid(),elapsed_seconds=time.time()-START,updated_unix=time.time(),**kw);write('status.json',d);print(json.dumps(d),flush=True)
def save(name,value):
 p=OUT/name;t=p.with_suffix('.tmp');torch.save(value,t);t.replace(p)
def augmentation():
 path=OUT/'train_baseline.npz'
 if path.exists():return dict(np.load(path))
 raw=np.load(OLD/'train_raw.npz');h0,n0=raw['truth'],raw['noise'];count=len(h0)
 rng=np.random.default_rng(SEED);peaks=np.argmax(abs(hilbert(h0.astype(float),axis=1)),axis=1)
 h=np.tile(h0,(6,1));n=np.empty_like(h);mask=np.zeros_like(h);starts=[];noise_ids=[]
 for view in range(6):
  perm=rng.permutation(count);n[view*count:(view+1)*count]=n0[perm];noise_ids.append(perm)
  s=rng.integers(132,264,count) if view%2==0 else np.clip(peaks-rng.integers(8,59,count),132,263)
  starts.append(s)
 starts=np.concatenate(starts)
 for i,s in enumerate(starts):mask[i,s:s+66]=1
 x=h*(1-mask)+n
 assert np.array_equal(x,h*(1-mask)+n)
 np.savez_compressed(OUT/'augmentation_manifest.npz',signal_ids=np.tile(np.arange(count),6),noise_ids=np.concatenate(noise_ids),gap_starts=starts,source_seeds=raw['seeds'])
 from model.QTranTimeMixerMod import QTransformModule
 from model.mymodel_5 import DenoiseMaskedGapsFiller
 base=DenoiseMaskedGapsFiller().cuda().eval();base.load_state_dict(torch.load(previous.BP,map_location='cuda',weights_only=True));q=QTransformModule();pred=np.empty_like(h);t=time.time()
 with torch.no_grad():
  for lo in range(0,len(h),32):
   v=torch.from_numpy(x[lo:lo+32]);qt=torch.stack([q(row) for row in v]).float().cuda();pred[lo:lo+32]=base(v.cuda(),qt).cpu().numpy()
   if lo%1024==0 or lo+32>=len(h):
    done=min(lo+32,len(h));rate=done/(time.time()-t);status('augmentation_baseline',completed=done,total=len(h),examples_per_second=rate,stage_remaining_seconds=(len(h)-done)/rate)
 del base;torch.cuda.empty_cache();data=dict(truth=h,observation=x,baseline=pred,mask=mask,gap_starts=starts)
 status('saving_augmented_cache');np.savez_compressed(path,**data);return data
def main():
 torch.set_num_threads(4);torch.manual_seed(SEED);np.random.seed(SEED)
 assert torch.cuda.is_available()
 config=dict(train_unique=12000,train_augmented_combinations=72000,validation_n=1500,new_final_test_n=1000,new_test_seed_start=2026240000,observation='G*h+n',max_epochs=MAX_EPOCHS,min_epochs=MIN_EPOCHS,patience=PATIENCE,min_delta=DELTA,cosine_cycle_epochs=200,lr_max=1e-4,lr_min=2e-6,seed=SEED,warmstart_sha256=digest(CP),baseline_sha256=digest(previous.BP),psd_sha256=digest(PSD),selection='0.75*validation mean + 0.25*peak-in-gap validation mean',test_policy='fresh test generated only after checkpoint selection; retain every row')
 write('config.json',config);status('starting',config=config)
 train=augmentation();val=dict(np.load(OLD/'validation_baseline.npz'))
 psd=np.load(PSD)['psd'].copy();psd[[0,-1]]=np.inf
 w=torch.tensor(np.median(psd[np.isfinite(psd)])/psd,device='cuda',dtype=torch.float32);w/=w.mean()
 state=torch.load(CP,map_location='cpu',weights_only=True);model=MaskAwareRefiner(state['width']).cuda();model.load_state_dict(state['state_dict'])
 def tensors(d):return [torch.tensor(d[k]*(1 if k=='mask' else 1e22),device='cuda') for k in ['observation','baseline','mask','truth']]
 tr=tensors(train);va=tensors(val)
 peaks=np.argmax(abs(hilbert(val['truth'].astype(float),axis=1)),axis=1);pg=val['mask'][np.arange(len(peaks)),peaks].astype(bool)
 def predict(data):
  model.eval();out=[]
  with torch.no_grad():
   for lo in range(0,len(data[0]),128):out.append(model(*[a[lo:lo+128] for a in data[:3]]).cpu().numpy().astype(float)/1e22)
  return np.concatenate(out)
 def evaluate():
  mm=mismatch(val['truth'].astype(float),predict(va),psd);a=float(mm.mean());b=float(mm[pg].mean());return .75*a+.25*b,a,b
 def lossfn(p,h,m):
  err=(p-h).square();relative=(err.sum(1)+2*(err*m).sum(1))/h.square().sum(1).clamp_min(1e-4)
  P,T=torch.fft.rfft(p,dim=1),torch.fft.rfft(h,dim=1);pp=(P.abs().square()*w).sum(1).clamp_min(1e-5);tt=(T.abs().square()*w).sum(1).clamp_min(1e-5)
  spec=((P-T).abs().square()*w).sum(1)/tt;ov=1-((P*T.conj()).real*w).sum(1)/(pp*tt).sqrt()
  return (relative+.5*spec+.25*ov).mean()
 best,a,b=evaluate();initial=best;best_epoch=0;stale=0;history=[];times=[]
 if not (OUT/'last_checkpoint.pth').exists():save('best_refiner.pth',dict(state_dict=copy.deepcopy(model.state_dict()),width=state['width'],epoch=0,score=best,input_scale=1e22))
 opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
 scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=200,T_mult=1,eta_min=2e-6)
 first=1
 if (OUT/'last_checkpoint.pth').exists():
  ck=torch.load(OUT/'last_checkpoint.pth',map_location='cuda',weights_only=False);model.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);best=ck['best'];best_epoch=ck['best_epoch'];stale=ck['stale'];first=ck['epoch']+1;torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state(ck['cuda_rng'].cpu());history=json.loads((OUT/'history.json').read_text())
 write('initial_validation.json',dict(score=initial,population_mean=a,peak_mean=b,peak_count=int(pg.sum())))
 for epoch in range(first,MAX_EPOCHS+1):
  t=time.time();model.train();order=torch.randperm(len(tr[0]),device='cuda');total=0
  for lo in range(0,len(order),128):
   ix=order[lo:lo+128];opt.zero_grad(set_to_none=True);loss=lossfn(model(tr[0][ix],tr[1][ix],tr[2][ix]),tr[3][ix],tr[2][ix]);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();total+=float(loss.detach())*len(ix)
  score,a,b=evaluate();assert np.isfinite(score)
  if score<best-DELTA:
   best=score;best_epoch=epoch;stale=0;save('best_refiner.pth',dict(state_dict=copy.deepcopy(model.state_dict()),width=state['width'],epoch=epoch,score=best,input_scale=1e22))
  else:stale+=1
  used_lr=opt.param_groups[0]['lr'];scheduler.step(epoch);secs=time.time()-t;times.append(secs);rate=float(np.median(times[-10:]));row=dict(epoch=epoch,train_loss=total/len(order),validation_score=score,population_mean=a,peak_mean=b,best_epoch=best_epoch,best_score=best,stale_epochs=stale,lr=used_lr,epoch_seconds=secs);history.append(row);write('history.json',history)
  save('last_checkpoint.pth',dict(state_dict=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,best=best,best_epoch=best_epoch,stale=stale,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()))
  status('training',**row,median_epoch_seconds=rate,max_training_remaining_seconds=(MAX_EPOCHS-epoch)*rate,early_stop_if_no_improvement_seconds=max(MIN_EPOCHS-epoch,PATIENCE-stale,0)*rate)
  if (epoch>=MIN_EPOCHS and stale>=PATIENCE) or (OUT/'STOP').exists():break
 selected=torch.load(OUT/'best_refiner.pth',map_location='cuda',weights_only=True);model.load_state_dict(selected['state_dict'])
 test=previous.cached('final_test',1000,2026240000);te=tensors(test);new=predict(te);model.load_state_dict(state['state_dict']);old=predict(te);model.load_state_dict(selected['state_dict'])
 h=test['truth'].astype(float);peaks=np.argmax(abs(hilbert(h,axis=1)),axis=1);groups=dict(all=np.ones(1000,bool),peak_in_gap=test['mask'][np.arange(1000),peaks].astype(bool));rng=np.random.default_rng(SEED);rows=[]
 for refname,ref in [('previous_best',old),('DGF14',test['baseline'].astype(float))]:
  a=mismatch(h,new,psd);b=mismatch(h,ref,psd)
  for name,keep in groups.items():
   delta=a[keep]-b[keep];ci=np.percentile(rng.choice(delta,size=(5000,len(delta)),replace=True).mean(1),[2.5,97.5]);rows.append(dict(comparison=refname,group=name,n=int(keep.sum()),reference=summary(b[keep]),augmented=summary(a[keep]),paired_mean_delta=float(delta.mean()),paired_bootstrap95ci=ci.tolist()))
 np.savez_compressed(OUT/'final_test_predictions.npz',augmented=new,previous=old,baseline=test['baseline']);write('final_results.json',dict(selected_epoch=selected['epoch'],rows=rows,config=config));status('completed',selected_epoch=selected['epoch'],finished_epoch=epoch,stop_reason='patience' if stale>=PATIENCE and epoch>=MIN_EPOCHS else ('manual_STOP' if (OUT/'STOP').exists() else 'max_epochs'),rows=rows)
if __name__=='__main__':
 try:main()
 except BaseException:
  status('failed',error=traceback.format_exc());raise
