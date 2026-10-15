from pathlib import Path
import numpy as np,json
from scipy.signal import periodogram
P=Path(__file__).resolve().parent;R=Path('/home/ljq/code/RingdownGapFilling_PRD');F=R/'FillingGap/results_prd_tail_20261015';T=F/'fresh_test';a=np.load(P/'publication_metrics.npz');t=np.arange(1056)*.5
h=np.load(T/'truth.npy',mmap_mode='r');n=np.load(T/'noise.npy',mmap_mode='r');pred=np.load(F/'variable_optimized_prediction.npy',mmap_mode='r');psd=np.load(R/'FillingGap/results_prd_physical_transfer_20261009/calibration_psd.npz')['psd'];asd=np.sqrt(psd);asd[:,0]=asd[:,1];asd[:,-1]=asd[:,-2]
sums={k:np.zeros((2,529)) for k in ['input_error','residual','noise']}
for lo in range(0,6000,128):
 hi=min(lo+128,6000);v=h[lo:hi].astype(float);nv=n[lo:hi].astype(float);pr=np.fft.irfft(np.fft.rfft(pred[lo:hi].astype(float))*asd,n=1056);m=(t>=a['start'][lo:hi,None])&(t<a['start'][lo:hi,None]+a['length'][lo:hi,None]);er=nv-v*m[:,None]
 for key,x in [('input_error',er),('residual',pr-v),('noise',nv)]:
  f,pp=periodogram(x,fs=2,window='boxcar',detrend='constant',scaling='density');sums[key]+=pp.sum(0)
sums={k:v/6000 for k,v in sums.items()};np.savez_compressed(P/'error_spectra.npz',frequency=f,**sums)
old=np.load(P/'population_spectra.npz');assert np.allclose(sums['noise'],old['noise'],rtol=1e-12,atol=0);assert np.allclose(sums['residual'],old['residual'],rtol=1e-12,atol=0)
weights=np.ones(len(f));weights[[0,-1]]=1
ratio=(sums['residual']*weights).sum(1)/(sums['input_error']*weights).sum(1)
# Detrending removes the per-channel DC offset; use the same operator for both errors.
report=dict(population=6000,input_error='d-h=n-m*h',reconstruction_error='prediction-h',periodogram='rectangular, constant detrend, one-sided density, arithmetic source mean',integrated_residual_to_input_power=ratio.tolist(),integrated_reduction_db=(-10*np.log10(ratio)).tolist(),all_interior_bins_lower=bool((sums['residual'][:,1:-1]<sums['input_error'][:,1:-1]).all()))
(P/'spectral_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
