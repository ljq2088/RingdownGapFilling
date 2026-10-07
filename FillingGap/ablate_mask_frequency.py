"""Matched-budget mask/frequency ablations. Validation only; no test access.
Uses the same 2048 unique training signals with both mask views for all variants.
Train from zero-head initialization; compare 3 seeds, never pick on final test.
"""
from pathlib import Path
import os,sys,json,time,argparse,hashlib,copy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.signal import hilbert
from model.mask_aware_refiner import MaskAwareRefiner
from FillingGap.compare_mismatch_metrics import mismatch
CACHE=ROOT/'FillingGap/results_longrun_20261008'
OUT=ROOT/'FillingGap/results_mask_frequency_ablation'
VARIANTS={'time_only':(False,False),'mask_only':(True,False),'frequency_only':(False,True),'mask_frequency':(True,True)}
SEEDS=[2026100821,2026100822,2026100823]
def lossfn(p,h,m,w,frequency):
 err=(p-h).square();relative=(err.sum(1)+2*(err*m).sum(1))/h.square().sum(1).clamp_min(1e-4)
 if not frequency:return relative.mean()
 P=torch.fft.rfft(p,dim=1);T=torch.fft.rfft(h,dim=1)
 pp=(P.abs().square()*w).sum(1).clamp_min(1e-5);tt=(T.abs().square()*w).sum(1).clamp_min(1e-5)
 spec=((P-T).abs().square()*w).sum(1)/tt
 ov=1-((P*T.conj()).real*w).sum(1)/(pp*tt).sqrt()
 return (relative+.5*spec+.25*ov).mean()
def atomic(name,data):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(p)
def prerequisites():
 p=CACHE/'pid.txt'
 if p.exists():
  proc=Path('/proc')/p.read_text().strip()
  if proc.exists() and b'train_longrun.py' in (proc/'cmdline').read_bytes():
   raise SystemExit('Primary long run is live; do not compete for its GPU.')
 status=json.loads((CACHE/'status.json').read_text())
 if status['stage']!='completed':raise SystemExit('Primary run must complete before ablations.')
 for name in ['train_baseline.npz','validation_baseline.npz']:assert (CACHE/name).exists()
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
 prerequisites()
 if args.check:print('Ablation prerequisites passed. No final test access.');return
 assert torch.cuda.is_available();torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
 cfg={'unique_training_signals':2048,'training_mask_views':4096,'max_epochs':100,'patience':20,'min_delta':1e-5,'batch_size':128,'seeds':SEEDS,'variants':VARIANTS,'initialization':'from scratch with zero output head','comparison':'same architecture, data, shuffle seed and training budget; no warm start','selection_metric':'0.75*natural_validation_mean+0.25*peak_in_gap_validation_mean','observation':'G*h+n','baseline':'frozen DGF14','test_access':False,'source_sha256':{name:hashlib.sha256((CACHE/name).read_bytes()).hexdigest() for name in ['train_baseline.npz','validation_baseline.npz']}}
 atomic('protocol.json',cfg)
 z=np.load(CACHE/'train_baseline.npz');v=np.load(CACHE/'validation_baseline.npz')
 count=len(z['truth'])//2;unique=np.random.default_rng(2026100820).choice(count,2048,replace=False);indices=np.concatenate([unique,unique+count])
 np.save(OUT/'training_cache_rows.npy',indices)
 def tensors(data,idx):return [torch.tensor(data[k][idx]*(1 if k=='mask' else 1e22),device='cuda',dtype=torch.float32) for k in ['observation','baseline','mask','truth']]
 tr=tensors(z,indices);va=tensors(v,np.arange(len(v['truth'])))
 psd=np.load(ROOT/'FillingGap/results_psd_reaudit/calibration_noise_psds.npz')['psd'].copy();psd[[0,-1]]=np.inf
 w=torch.tensor(np.median(psd[np.isfinite(psd)])/psd,device='cuda',dtype=torch.float32);w/=w.mean()
 peak=np.argmax(abs(hilbert(v['truth'].astype(float),axis=1)),axis=1);pg=v['mask'][np.arange(len(peak)),peak].astype(bool);assert pg.any()
 results=[]
 for seed in SEEDS:
  for name,(use_mask,frequency) in VARIANTS.items():
   cp=OUT/f'{name}_{seed}.pth';summary_path=OUT/f'{name}_{seed}.json'
   if summary_path.exists():results.append(json.loads(summary_path.read_text()));continue
   torch.manual_seed(seed);model=MaskAwareRefiner().cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
   scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=6,threshold=1e-5,threshold_mode='abs',min_lr=1e-6)
   def forward(data,idx):return model(data[0][idx],data[1][idx],data[2][idx] if use_mask else torch.zeros_like(data[2][idx]))
   def evaluate():
    model.eval();pred=[]
    with torch.no_grad():
     for lo in range(0,len(va[0]),128):pred.append(forward(va,slice(lo,lo+128)).cpu().numpy().astype(float)/1e22)
    pred=np.concatenate(pred);mm=mismatch(v['truth'].astype(float),pred,psd);a=float(mm.mean());b=float(mm[pg].mean());return .75*a+.25*b,a,b,pred
   best,a,b,pred=evaluate();best_epoch=0;best_state=copy.deepcopy(model.state_dict());stale=0;history=[];started=time.time()
   # The shuffle generator is separate: mask/frequency paths get identical orders.
   g=torch.Generator(device='cuda');g.manual_seed(seed)
   for epoch in range(1,101):
    model.train();order=torch.randperm(len(tr[0]),device='cuda',generator=g);total=0
    for lo in range(0,len(order),128):
     idx=order[lo:lo+128];opt.zero_grad(set_to_none=True);loss=lossfn(forward(tr,idx),tr[3][idx],tr[2][idx],w,frequency);assert torch.isfinite(loss)
     loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();total+=float(loss.detach())*len(idx)
    score,a,b,pred=evaluate();assert np.isfinite(score);scheduler.step(score)
    if score<best-1e-5:best=score;best_epoch=epoch;stale=0;best_state=copy.deepcopy(model.state_dict())
    else:stale+=1
    history.append({'epoch':epoch,'train_loss':total/len(order),'score':score,'population_mean':a,'peak_mean':b,'stale':stale})
    atomic('status.json',{'variant':name,'seed':seed,'epoch':epoch,'best_epoch':best_epoch,'score':score,'elapsed_seconds':time.time()-started})
    print(name,seed,epoch,score,flush=True)
    if stale>=20:break
   model.load_state_dict(best_state);score,a,b,pred=evaluate();torch.save({'state_dict':best_state,'width':24,'epoch':best_epoch,'seed':seed,'variant':name,'mask_enabled':use_mask,'input_scale':1e22},cp)
   row={'variant':name,'seed':seed,'score':score,'population_mean':a,'peak_mean':b,'best_epoch':best_epoch,'finished_epoch':epoch,'history':history};atomic(summary_path.name,row);results.append(row)
   np.savez_compressed(OUT/f'{name}_{seed}_validation_predictions.npz',pred=pred)
 means={name:float(np.mean([r['score'] for r in results if r['variant']==name])) for name in VARIANTS}
 atomic('results.json',{'rows':results,'mean_score_by_variant':means,'best_variant_by_mean_validation_score':min(means,key=means.get),'not_a_test_result':True})
if __name__=='__main__':main()
