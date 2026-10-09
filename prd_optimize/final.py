"""Only after validation choices freeze, generate a NEW independent physical test set."""
import os
os.environ.setdefault("OMP_NUM_THREADS","2");os.environ.setdefault("NUMBA_NUM_THREADS","1")
import json,time
import numpy as np,torch
from prd_physical.common import N,Preprocess,metric,summary,write,sha
from prd_physical.physics import Simulator
from prd_physical.prepare import crop_meta
from model.QTranTimeMixerMod import QTransformModule
from .settings import *
from .model import OptimizedDGF
from .data import batches,predict
from .train import check_sources

def status(stage,**kw):
 v=dict(stage=stage,pid=os.getpid(),updated_unix=time.time(),**kw);write(OUT/"status.json",v);print(json.dumps(v),flush=True)
def main():
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(GPU_FRACTION);check_sources()
 frozen=json.loads((OUT/"selection_frozen.json").read_text());assert frozen["new_test_read"] is False
 selections={s["variant"]:s for s in frozen["candidates"]}
 for kind in VARIANTS:assert sha(OUT/kind/"best.pth")==selections[kind]["checkpoint_sha256"]
 test=OUT/"fresh_test.npz"
 if not test.exists():
  sim=Simulator();h=[];noise=[];masks=[];params=[];t=time.monotonic()
  for i in range(TEST_N):
   seed=TEST_SEED+i;full,n,par=sim.example(seed);start,m=crop_meta(par,seed)
   h.append(full[:,start:start+N]);noise.append(n[:,start:start+N]);masks.append(m);params.append(par)
   if (i+1)%50==0:status("generating_fresh_test",completed=i+1,total=TEST_N,remaining_seconds=(TEST_N-i-1)*(time.monotonic()-t)/(i+1))
  tmp=test.with_name("fresh_test.tmp.npz")
  np.savez_compressed(tmp,truth=np.stack(h),noise=np.stack(noise),mask=np.stack(masks),parameters=np.stack(params),seeds=np.arange(TEST_SEED,TEST_SEED+TEST_N))
  tmp.replace(test)
 with np.load(test) as f:h=f["truth"].astype(float);noise=f["noise"];masks=f["mask"];params=f["parameters"];seeds=f["seeds"]
 assert len(h)==TEST_N and np.array_equal(seeds,np.arange(TEST_SEED,TEST_SEED+TEST_N))
 for split in ["train","validation"]:
  old=np.load(OUT/"cache"/(split+"_seeds.npy"),mmap_mode="r");assert not np.isin(seeds,old).any()
 assert (masks.sum(-1)==66).all()
 x=h*(1-masks[:,None,:])+noise;cal=np.load(OLD/"calibration_psd.npz");pre=Preprocess(cal["psd"],"white");u=pre.encode(x)
 q=QTransformModule();scale=np.sqrt((u.astype(float)**2).mean(-1)).clip(1e-8);qt=[]
 for i,(row,sc) in enumerate(zip(u,scale)):
  qt.append(torch.stack([q(torch.from_numpy(channel/s)) for channel,s in zip(row,sc)]).numpy())
  if (i+1)%1000==0:status("fresh_test_features",completed=i+1,total=TEST_N)
 a=dict(input=torch.from_numpy(u),target=torch.zeros_like(torch.from_numpy(u)),q=torch.from_numpy(np.stack(qt)),mask=torch.from_numpy(masks))
 model=OptimizedDGF("continuation").cuda().eval()
 with torch.no_grad():a["bands"]=torch.cat([model.input_bands(a["input"][lo:lo+BATCH].cuda()).cpu() for lo in range(0,TEST_N,BATCH)])
 preds={"previous_white":predict(model,a,pre)};del model;torch.cuda.empty_cache()
 for kind in VARIANTS:
  status("evaluating_fresh_test",variant=kind);model=OptimizedDGF(kind).cuda();ck=torch.load(OUT/kind/"best.pth",map_location="cuda",weights_only=True);model.load_state_dict(ck["state_dict"])
  preds[kind]=predict(model,a,pre);del model;torch.cuda.empty_cache()
 peaks=np.argmax((h*h).sum(1),axis=-1);pg=masks[np.arange(TEST_N),peaks].astype(bool);per={};reports={};hann=np.hanning(N)
 for name,pred in preds.items():
  mm=metric(h,pred,cal["psd"]);hm=metric(h*hann,pred*hann,cal["hann_psd"]);tm=metric(h,pred);rel=np.sqrt(((pred-h)**2).sum((1,2))/(h*h).sum((1,2)))
  for met,val in [("psd",mm),("hann_psd",hm),("time",tm),("relative_l2",rel)]:per[name+"__"+met]=val
  reports[name]=dict(psd=summary(mm),hann_psd=summary(hm),time=summary(tm),relative_l2=summary(rel),peak_in_gap_psd=summary(mm[pg]))
 preferred=frozen["preferred"];paired={};rng=np.random.default_rng(SEED+3)
 for ref in reports:
  if ref==preferred:continue
  d=per[preferred+"__psd"]-per[ref+"__psd"];bs=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(5000)])
  paired[ref]=dict(mean_delta=float(d.mean()),ci95=np.quantile(bs,[.025,.975]).tolist())
 result=dict(preferred_from_validation=preferred,n=TEST_N,all_rows_retained=True,observation="G*h+n",time_phase_maximized=False,new_independent_test=True,seeds=[TEST_SEED,TEST_SEED+TEST_N-1],results=reports,paired=paired,selected=frozen,improved_vs_previous_on_same_new_test=(preferred!="previous_white" and paired["previous_white"]["ci95"][1]<0),historical_legacy_mean=.005368768966470572,cross_distribution_comparison="historical mean contextual only",scope="528-s ideal equal-arm physical TDI A/E synthetic ringdown segments; source bootstrap conditional on checkpoints, not training-seed uncertainty")
 np.savez_compressed(OUT/"per_example_test.npz",seeds=seeds,parameters=params,peak_in_gap=pg,**per)
 np.savez_compressed(OUT/"test_predictions.npz",truth=h,noise=noise,mask=masks,**preds)
 # Independent CPU check in whitened Fourier coordinates before reporting completion.
 errors={}
 for name,pred in preds.items():
  H=np.fft.rfft(h,axis=-1)[...,1:-1]/np.sqrt(cal["psd"][:,1:-1])
  P=np.fft.rfft(pred,axis=-1)[...,1:-1]/np.sqrt(cal["psd"][:,1:-1])
  cross=(H.conj()*P).real.sum((1,2));den=np.sqrt((abs(H)**2).sum((1,2))*(abs(P)**2).sum((1,2)))
  direct=1-np.clip(np.divide(cross,den,out=np.zeros(TEST_N),where=den>0),-1,1)
  error=float(np.max(abs(direct-per[name+"__psd"])));assert error<1e-12
  errors[name]=error
 write(OUT/"final_metric_audit.json",dict(passed=True,n=TEST_N,independent_psd_max_errors=errors,test_disjoint_from_train_validation=True,all_rows_retained=True,checkpoints_hash_verified=True))
 write(OUT/"final_results.json",result);status("completed",preferred=preferred,n=TEST_N,improved_same_test=result["improved_vs_previous_on_same_new_test"],psd_mean=reports[preferred]["psd"]["mean"])
if __name__=="__main__":main()
