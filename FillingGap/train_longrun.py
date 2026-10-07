"""Detached mask-aware refinement run with new train/validation/test seeds."""
import os
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('NUMBA_NUM_THREADS','1')
from pathlib import Path
import sys,json,time,traceback,multiprocessing as mp,copy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.signal import hilbert
from model.mask_aware_refiner import MaskAwareRefiner
from FillingGap.compare_mismatch_metrics import mismatch,summary,digest
OUT=ROOT/'FillingGap/results_longrun_20261008';OUT.mkdir(exist_ok=True)
TRAIN_N=12000;VAL_N=1500;TEST_N=1000;MAX_EPOCHS=600;PATIENCE=50;MIN_DELTA=1e-5
BP=ROOT/'saved_models/noise/lowSNR/model_with_noise_gaps_HugeRangeParas_14.pth'
CP=ROOT/'FillingGap/results_model_optimization/best_refiner.pth'
PSD_PATH=ROOT/'FillingGap/results_psd_reaudit/calibration_noise_psds.npz'
START=time.time()
def atomic_json(name,value):
 p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)
def status(stage,**kw):
 d={'stage':stage,'pid':os.getpid(),'updated_unix':time.time(),'elapsed_seconds':time.time()-START,**kw}
 atomic_json('status.json',d);print(json.dumps(d),flush=True)
def save_checkpoint(name,data):
 p=OUT/name;t=p.with_suffix('.tmp');torch.save(data,t);t.replace(p)
def generate(seed):
 from data.data_generation import generate_single_data
 h,n,_,_=generate_single_data(seed,seed=seed)
 return h.numpy().astype(np.float32)*.03,n.numpy().astype(np.float32)
def raw_data(name,count,seed):
 path=OUT/(name+'_raw.npz')
 if path.exists():
  z=np.load(path);return z['truth'],z['noise']
 h=np.empty((count,1056),np.float32);n=np.empty_like(h);t=time.time()
 with mp.get_context('spawn').Pool(4) as pool:
  for i,(a,b) in enumerate(pool.imap(generate,range(seed,seed+count),chunksize=8)):
   h[i]=a;n[i]=b
   if (i+1)%128==0 or i+1==count:
    rate=(i+1)/(time.time()-t);status('generate_'+name,completed=i+1,total=count,examples_per_second=rate,stage_remaining_seconds=(count-i-1)/rate)
 np.savez_compressed(path,truth=h,noise=n,seeds=np.arange(seed,seed+count));return h,n
def cached(name,count,seed,two_masks=False):
 path=OUT/(name+'_baseline.npz')
 if path.exists():return dict(np.load(path))
 h,n=raw_data(name,count,seed);rng=np.random.default_rng(seed);starts=rng.integers(132,264,count)
 if two_masks:
  peak=np.argmax(abs(hilbert(h.astype(float),axis=1)),axis=1)
  targeted=np.clip(peak-rng.integers(8,59,count),132,263)
  h=np.concatenate([h,h]);n=np.concatenate([n,n]);starts=np.concatenate([starts,targeted])
 mask=np.zeros_like(h,dtype=np.float32)
 for i,s in enumerate(starts):mask[i,s:s+66]=1
 x=h*(1-mask)+n
 from model.QTranTimeMixerMod import QTransformModule
 from model.mymodel_5 import DenoiseMaskedGapsFiller
 base=DenoiseMaskedGapsFiller().cuda().eval();base.load_state_dict(torch.load(BP,map_location='cuda',weights_only=True));q=QTransformModule()
 pred=np.empty_like(h);t=time.time()
 with torch.no_grad():
  for lo in range(0,len(h),32):
   v=torch.from_numpy(x[lo:lo+32]);qt=torch.stack([q(row) for row in v]).float().cuda()
   pred[lo:lo+32]=base(v.cuda(),qt).cpu().numpy()
   if lo%256==0 or lo+32>=len(h):
    done=min(lo+32,len(h));rate=done/(time.time()-t);status('baseline_'+name,completed=done,total=len(h),examples_per_second=rate,stage_remaining_seconds=(len(h)-done)/rate)
 del base;torch.cuda.empty_cache()
 result={'truth':h,'observation':x,'baseline':pred,'mask':mask,'gap_starts':starts}
 np.savez_compressed(path,**result);return result
def main():
 torch.set_num_threads(4);torch.manual_seed(2026100811);np.random.seed(2026100811)
 assert torch.cuda.is_available(),'CUDA required for this long run'
 cfg={'train_unique':TRAIN_N,'train_masked_examples':2*TRAIN_N,'validation_natural':VAL_N,'final_test':TEST_N,'train_seed_start':2026101000,'validation_seed_start':2026121000,'test_seed_start':2026131000,'max_epochs':MAX_EPOCHS,'patience':PATIENCE,'min_delta':MIN_DELTA,'batch_size':128,'learning_rate':3e-4,'validation_score':'0.75*population_mean_PSD_mismatch + 0.25*peak_in_gap_mean_PSD_mismatch','observation':'G*h+n','baseline_sha256':digest(BP),'warmstart_sha256':digest(CP),'psd_sha256':digest(PSD_PATH),'test_policy':'test generated/evaluated once only after early stopping; no exclusions','hardware':torch.cuda.get_device_name()}
 atomic_json('config.json',cfg);status('starting',config=cfg)
 train=cached('train',TRAIN_N,2026101000,True);val=cached('validation',VAL_N,2026121000)
 psd=np.load(PSD_PATH)['psd'].copy();psd[[0,-1]]=np.inf
 weight=torch.tensor(np.median(psd[np.isfinite(psd)])/psd,device='cuda',dtype=torch.float32);weight/=weight.mean()
 state=torch.load(CP,map_location='cpu',weights_only=True);model=MaskAwareRefiner(state['width']).cuda();model.load_state_dict(state['state_dict'])
 def tensors(data):return [torch.tensor(data[k]* (1 if k=='mask' else 1e22),device='cuda',dtype=torch.float32) for k in ['observation','baseline','mask','truth']]
 tr=tensors(train);va=tensors(val)
 peaks=np.argmax(abs(hilbert(val['truth'].astype(float),axis=1)),axis=1);pg=val['mask'][np.arange(VAL_N),peaks].astype(bool)
 assert pg.sum()>0
 def lossfn(p,h,m):
  den=h.square().sum(1).clamp_min(1e-4);err=(p-h).square();relative=(err.sum(1)+2*(err*m).sum(1))/den
  P=torch.fft.rfft(p,dim=1);T=torch.fft.rfft(h,dim=1);pp=(P.abs().square()*weight).sum(1).clamp_min(1e-5);tt=(T.abs().square()*weight).sum(1).clamp_min(1e-5)
  spec=((P-T).abs().square()*weight).sum(1)/tt;ov=1-((P*T.conj()).real*weight).sum(1)/(pp*tt).sqrt()
  return (relative+.5*spec+.25*ov).mean()
 def predict(data):
  model.eval();out=[]
  with torch.no_grad():
   for lo in range(0,len(data[0]),128):out.append(model(*[a[lo:lo+128] for a in data[:3]]).cpu().numpy().astype(float)/1e22)
  return np.concatenate(out)
 def evaluate():
  m=mismatch(val['truth'].astype(float),predict(va),psd);a=float(m.mean());b=float(m[pg].mean());return .75*a+.25*b,a,b
 opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
 sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=12,threshold=MIN_DELTA,threshold_mode='abs',min_lr=1e-6)
 best,a,b=evaluate();best_epoch=0;stale=0;first_epoch=1;history=[];epoch_times=[]
 if not (OUT/'last_checkpoint.pth').exists():save_checkpoint('best_refiner.pth',{'state_dict':copy.deepcopy(model.state_dict()),'epoch':0,'width':state['width'],'score':best,'input_scale':1e22})
 if (OUT/'last_checkpoint.pth').exists():
  ck=torch.load(OUT/'last_checkpoint.pth',map_location='cuda',weights_only=False);model.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer']);sch.load_state_dict(ck['scheduler']);best=ck['best'];best_epoch=ck['best_epoch'];stale=ck['stale'];first_epoch=ck['epoch']+1;torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state(ck['cuda_rng'].cpu());np.random.set_state(ck['numpy_rng']);history=json.loads((OUT/'history.json').read_text())
 status('training',epoch=first_epoch-1,validation_score=best,population_mean=a,peak_mean=b,peak_count=int(pg.sum()))
 for epoch in range(first_epoch,MAX_EPOCHS+1):
  t=time.time();model.train();order=torch.randperm(len(tr[0]),device='cuda');total=0
  for lo in range(0,len(order),128):
   ix=order[lo:lo+128];opt.zero_grad(set_to_none=True);loss=lossfn(model(tr[0][ix],tr[1][ix],tr[2][ix]),tr[3][ix],tr[2][ix]);assert torch.isfinite(loss),'Nonfinite loss'
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();total+=float(loss.detach())*len(ix)
  score,a,b=evaluate();assert np.isfinite(score);sch.step(score)
  if score<best-MIN_DELTA:
   best=score;best_epoch=epoch;stale=0;save_checkpoint('best_refiner.pth',{'state_dict':copy.deepcopy(model.state_dict()),'epoch':epoch,'width':state['width'],'score':score,'input_scale':1e22})
  else:stale+=1
  seconds=time.time()-t;epoch_times.append(seconds);meansec=float(np.median(epoch_times[-10:]));row={'epoch':epoch,'train_loss':total/len(order),'validation_score':score,'population_mean':a,'peak_mean':b,'best_epoch':best_epoch,'stale_epochs':stale,'lr':opt.param_groups[0]['lr'],'epoch_seconds':seconds};history.append(row);atomic_json('history.json',history)
  save_checkpoint('last_checkpoint.pth',{'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sch.state_dict(),'epoch':epoch,'best':best,'best_epoch':best_epoch,'stale':stale,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'numpy_rng':np.random.get_state()})
  status('training',**row,median_epoch_seconds=meansec,max_training_remaining_seconds=(MAX_EPOCHS-epoch)*meansec,early_stop_if_no_further_improvement_seconds=max(0,PATIENCE-stale)*meansec)
  if stale>=PATIENCE or (OUT/'STOP').exists():break
 # The test does not affect early stopping, learning rate or checkpoint selection.
 selected=torch.load(OUT/'best_refiner.pth',map_location='cuda',weights_only=True);model.load_state_dict(selected['state_dict'])
 test=cached('final_test',TEST_N,2026131000);p=predict(tensors(test));h=test['truth'].astype(float);baseline=test['baseline'].astype(float)
 prior=MaskAwareRefiner(state['width']).cuda().eval();prior.load_state_dict(state['state_dict']);current=model;model=prior;previous=predict(tensors(test));model=current
 peak=np.argmax(abs(hilbert(h,axis=1)),axis=1);groups={'all':np.ones(TEST_N,bool),'peak_in_gap':test['mask'][np.arange(TEST_N),peak].astype(bool)};rng=np.random.default_rng(2026131000);results=[]
 for name,ref in [('DGF14',baseline),('previous_short_refiner',previous)]:
  bm=mismatch(h,ref,psd);am=mismatch(h,p,psd)
  for group,keep in groups.items():
   delta=am[keep]-bm[keep];ci=np.percentile(rng.choice(delta,size=(2000,len(delta)),replace=True).mean(1),[2.5,97.5]);results.append({'comparison':name,'group':group,'n':int(keep.sum()),'reference':summary(bm[keep]),'longrun':summary(am[keep]),'paired_mean_delta':float(delta.mean()),'paired_bootstrap95ci':ci.tolist()})
 np.savez_compressed(OUT/'final_test_predictions.npz',longrun=p,previous=previous,baseline=baseline);atomic_json('final_results.json',{'selected_epoch':selected['epoch'],'rows':results,'config':cfg});status('completed',best_epoch=best_epoch,finished_epoch=epoch,stop_reason='patience' if stale>=PATIENCE else ('manual_STOP' if (OUT/'STOP').exists() else 'max_epochs'),results=results)
if __name__=='__main__':
 try:main()
 except BaseException:
  err=traceback.format_exc();status('failed',error=err);raise
