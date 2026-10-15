from pathlib import Path
import numpy as np,json
from scipy.signal import hilbert
from pycbc.conversions import get_lm_f0tau
P=Path(__file__).resolve().parent;R=Path('/home/ljq/code/RingdownGapFilling_PRD');T=R/'FillingGap/results_prd_tail_20261015/fresh_test'
a=np.load(P/'publication_metrics.npz');pars=np.load(T/'parameters.npy',mmap_mode='r');h=np.load(T/'truth.npy',mmap_mode='r');out=[]
for i in [3874,4884,4076]:
 f,tau=get_lm_f0tau(float(pars[i,9]),float(pars[i,10]),2,2);v=h[i].astype(float);peak=.5*np.argmax(np.sum(abs(hilbert(v))**2,axis=0));lo=max(0,np.floor((peak-1.8*tau)/.5)*.5);hi=min(527.5,np.floor((peak+4*tau)/.5)*.5);stop=min(527.5,np.floor((peak+3*tau)/.5)*.5)
 out.append(dict(index=i,**{k:a[k][i].tolist() for k in a.files},f220_hz=float(f),tau220_s=float(tau),peak_s=float(peak),detail_limits_s=[lo,hi],log_limits_s=[lo,stop],amplitude_scale=1e-18,detail_y_limit=float(1.2*abs(v).max()/1e-18)))
(P/'selected_examples.json').write_text(json.dumps(dict(selected=out,display_processing='Native samples; physical fractional-frequency units; original noise retained in full records and zooms. No display smoothing or relative alignment.',selection='Three rho>=10 examples with small waveform and gap errors, including a 31.5 mHz fundamental mode. Full-record statistics retain all 6000 sources.',log_window='From max(0, t_peak-1.8*tau220) to min(527.5, t_peak+3*tau220), rounded down to native 0.5-s sample times. Both channels use the joint clean analytic-envelope peak. Window uses clean truth and source parameters only.'),indent=2))
print(json.dumps(out,indent=2))
