from pathlib import Path
import json,sys,numpy as np
from scipy.signal import hilbert
ROOT=Path('/home/ljq/code/RingdownGapFilling_PRD');sys.path.insert(0,str(ROOT))
from prd_conditioned.settings import OUT,CAL
from prd_physical.common import metric
from pycbc.conversions import get_lm_f0tau
HERE=Path(__file__).resolve().parent
data=np.load(HERE/'diagnostic_native_samples.npz');stats=np.load(OUT/'per_example_test.npz')
parameters=np.load(OUT/'fresh_test/parameters.npy',mmap_mode='r')
rows=[]
for k,i in enumerate(data['index']):
 h=data['truth'][k];p=data['reconstruction'][k];r=p-h;m=data['mask'][k]
 z=hilbert(h);w=hilbert(p);env=abs(z);peak=env.argmax(-1)
 select=(np.arange(1056)[None,:]>peak[:,None])&(env>=.01*env.max(-1,keepdims=True))&(env<.1*env.max(-1,keepdims=True))
 f,tau=get_lm_f0tau(parameters[i,9],parameters[i,10],2,2)
 row={'index':int(i),'seed':int(data['seeds'][k]),'rank':[3000,5400,6000][k],
 'snr':float(stats['snr'][i]),'mismatch':float(data['mismatch'][k]),
 'f220_hz':float(f),'period220_seconds':float(1/f),'samples_per_period220':float(2/f),
 'global_relative_error':float(np.linalg.norm(r)/np.linalg.norm(h)),
 'gap_relative_error':float(np.linalg.norm(r*m)/np.linalg.norm(h*m)),
 'native_second_difference_peak_true_per_channel':(abs(np.diff(h,n=2)).max(-1)/abs(h).max(-1)).tolist(),
 'native_second_difference_peak_pred_per_channel':(abs(np.diff(p,n=2)).max(-1)/abs(h).max(-1)).tolist(),
 'residual_second_difference_norm_over_truth_per_channel':(np.linalg.norm(np.diff(r,n=2),axis=-1)/np.linalg.norm(np.diff(h,n=2),axis=-1)).tolist(),
 'initial_30s_residual_rms_over_channel_peak':(np.sqrt((r[:,:60]**2).mean(-1))/abs(h).max(-1)).tolist(),
 'tail_1_to_10_percent_relative_error':float(np.sqrt((r**2*select).sum()/(h**2*select).sum())),
 'tail_1_to_10_percent_phase_rms_degrees':float(np.sqrt((np.angle(w*z.conj())**2*env**2*select).sum()/(env**2*select).sum())*180/np.pi)}
 rows.append(row)
report={'cases':rows,'diagnostic_only':True,'figure_checkpoint_unchanged':True}
base=ROOT/'FillingGap/results_prd_tail_20261015'
status=json.loads((base/'status.json').read_text());report['training_status']=status
hp=base/'full/tail_unet/history.json'
hist=json.loads(hp.read_text());best=hist[-1]['best']
for r in hist:
 for name in ['raw','ema']:
  if abs(r[name]['score']-best)<1e-12:report['current_best_validation']={'epoch':r['epoch'],'weights':name,**r[name]}
(HERE/'figure8_smoothness.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
