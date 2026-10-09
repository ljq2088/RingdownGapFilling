import os
os.environ.setdefault("OMP_NUM_THREADS","2")
import json,time
from pathlib import Path
import numpy as np,torch,h5py
from prd_physical.common import metric,summary,Preprocess,sha,write,N
from prd_physical.model import PhysicalDGF
from prd_physical.run import BASE,load,predict
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/"FillingGap/results_prd_physical_transfer_20261009"
OUT=ROOT/"FillingGap/results_prd_optimization_20261010"
def main():
 OUT.mkdir(exist_ok=True);torch.set_num_threads(2)
 ck=OLD/"white/best.pth";cal=np.load(OLD/"calibration_psd.npz");pre=Preprocess(cal["psd"],"white")
 model=PhysicalDGF(BASE).cuda();model.load_state_dict(torch.load(ck,map_location="cuda",weights_only=True)["state_dict"])
 a=load("validation","white");p=predict(model,a,pre);h=a["physical_truth"];hann=np.hanning(N)
 with h5py.File(OLD/"validation_crops.h5","r") as f: pars=f["parameters"][:];mask=f["mask"][:];seeds=f["seeds"][:]
 psd=metric(h,p,cal["psd"]);hm=metric(h*hann,p*hann,cal["hann_psd"]);tm=metric(h,p)
 hf=np.fft.rfft(h,axis=-1);snr=np.sqrt((4/(2*N)*(abs(hf[...,1:-1])**2/cal["psd"][...,1:-1])).sum((1,2)))
 def buckets(values,edges):
  rows=[]
  for lo,hi in zip(edges[:-1],edges[1:]):
   ix=(values>=lo)&(values<hi)
   if ix.any():rows.append(dict(low=lo,high=hi,n=int(ix.sum()),psd=summary(psd[ix]),hann=summary(hm[ix])))
  return rows
 # Signed component contributions to overlap, not per-time PSD localization.
 e=((p-h)**2).sum(1);energy=(h*h).sum(1);bins=[]
 for lo,hi in [(0,132),(132,264),(264,528),(528,792),(792,1056)]:
  bins.append(dict(start=lo,end=hi,target_energy_fraction=float(energy[:,lo:hi].sum()/energy.sum()),error_energy_fraction=float(e[:,lo:hi].sum()/e.sum())))
 rep=dict(passed=True,n=len(h),checkpoint_sha256=sha(ck),validation_psd=summary(psd),validation_hann=summary(hm),validation_time=summary(tm),snr=buckets(snr,[0,10,20,50,100,1000,1e10]),source_mass=buckets(pars[:,0],[1e5,3e5,1e6,3e6,5e6+1]),damping_time=buckets(pars[:,12],[0,50,100,200,500,2000]),error_by_time=bins,hann_target_energy_ratio_median=float(np.median((energy*hann**2).sum(1)/energy.sum(1))),scope="validation only; no new test used for tuning")
 write(OUT/"validation_diagnosis.json",rep)
 np.savez_compressed(OUT/"previous_validation_predictions.npz",truth=h,prediction=p,mask=mask,seeds=seeds,psd_mismatch=psd,hann_mismatch=hm,snr=snr)
 print(json.dumps(rep),flush=True)
if __name__=="__main__":main()
