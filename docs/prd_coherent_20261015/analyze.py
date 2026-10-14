from pathlib import Path
import json,sys
import numpy as np
import torch
from scipy.signal import hilbert

ROOT=Path('/home/ljq/code/RingdownGapFilling_PRD');sys.path.insert(0,str(ROOT))
from prd_conditioned.settings import OUT as OLD, CACHE, CAL, PROBE
from prd_conditioned.model import ConditionedDGF
from prd_conditioned.data import MixedData
from prd_physical.common import Preprocess,metric,summary,N,FS
HERE=Path(__file__).resolve().parent
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.78)
pre=Preprocess(np.load(CAL)['psd'],'white')
def phase(h,p):
    a=hilbert(h,axis=-1);b=hilbert(p,axis=-1);j=np.abs(a).argmax(-1)
    ix=np.arange(len(h))[:,None];c=np.arange(2)[None,:]
    return np.abs(np.angle(b[ix,c,j]*a[ix,c,j].conj()))*180/np.pi
def channel_snr(h):
    H=np.fft.rfft(h)[...,1:-1]
    return np.sqrt((4/(FS*N)*np.abs(H)**2/np.load(CAL)['psd'][:,1:-1]).sum(-1))

pub=np.load(ROOT/'../RingdownGapFilling_PRD/FillingGap/results_prd_conditioned_20261014/per_example_test.npz')
fresh=OLD/'fresh_test';h=np.load(fresh/'truth.npy').astype(float)
p=pre.decode(np.load(OLD/'variable_optimized_prediction.npy'));err=phase(h,p);mm=pub['variable__optimized'];rho=pub['snr'];cr=channel_snr(h)
low=rho<10
report={'published_test_diagnostics_only':True,'new_model_selection_uses_validation_only':True,
 'low_snr':{'n':int(low.sum()),'fraction':float(low.mean()),'mean_mismatch':float(mm[low].mean()),'fraction_of_mismatch_sum':float(mm[low].sum()/mm.sum()),'phase_A':float(err[low,0].mean()),'phase_E':float(err[low,1].mean())},
 'channel_snr_phase_bins':[],'network_snr_phase_bins':[]}
for lo,hi in zip([0,5,10,20,50,100],[5,10,20,50,100,np.inf]):
    x=(cr>=lo)&(cr<hi); y=(rho>=lo)&(rho<hi)
    report['channel_snr_phase_bins'].append({'low':lo,'high':None if np.isinf(hi) else hi,'channel_values':int(x.sum()),'mean_abs_phase':float(err[x].mean()) if x.any() else None})
    report['network_snr_phase_bins'].append({'low':lo,'high':None if np.isinf(hi) else hi,'n':int(y.sum()),'mismatch':float(mm[y].mean()) if y.any() else None,'phase_A':float(err[y,0].mean()) if y.any() else None,'phase_E':float(err[y,1].mean()) if y.any() else None})
parameters=np.load(fresh/'parameters.npy')
report['source_mass_bins']=[]
for lo,hi in [(1e5,3e5),(3e5,1e6),(1e6,3e6),(3e6,5.000001e6)]:
    x=(parameters[:,0]>=lo)&(parameters[:,0]<hi)
    report['source_mass_bins'].append(dict(low=lo,high=hi,n=int(x.sum()),mean_mismatch=float(mm[x].mean()),phase_A=float(err[x,0].mean()),phase_E=float(err[x,1].mean())))
del h,p,err

h=np.load(CACHE/'validation_truth.npy').astype(float)
u=pre.encode(h).astype(float);f=np.fft.rfft(u);f[...,0]=f[...,-1]=0;u0=np.fft.irfft(f,n=N)
B=np.load(PROBE/'training_direction_basis.npz')['vectors'][:,:128].astype(float)
oracle=(u0.reshape(len(u0),-1)@B@B.T).reshape(u0.shape)+(u-u0)
physical=pre.decode(oracle)
report['validation_basis_oracle']={'rank':128,'noise_weighted_mismatch':summary(metric(h,physical,np.load(CAL)['psd'])),'phase_A':summary(phase(h,physical)[:,0]),'phase_E':summary(phase(h,physical)[:,1]),'basis_orthogonality_max_error':float(np.abs(B.T@B-np.eye(128)).max())}
model=ConditionedDGF('noise_phase').cuda().eval();model.load_state_dict(torch.load(OLD/'finetune/best.pth',map_location='cuda',weights_only=True)['state_dict'])
data=MixedData('validation');data.cache();out=[]
with torch.inference_mode():
    for lo in range(0,1500,128):
        v=data.batch(np.arange(lo,min(lo+128,1500)))
        out.append(model(v['input'],v['q'],v['mask'],v['bands'],v['q_low']).cpu().numpy())
pred=np.concatenate(out);np.save(HERE/'validation_baseline_white.npy',pred)
physical=pre.decode(pred);pe=phase(h,physical)
report['validation_baseline']={'noise_weighted_mismatch':summary(metric(h,physical,np.load(CAL)['psd'])),'phase_A':summary(pe[:,0]),'phase_E':summary(pe[:,1])}
encoder=model.backbone.backbone.dgf.backbone.dgf.base.encoder if hasattr(model,'backbone') and hasattr(model.backbone,'backbone') else model.backbone.dgf.backbone.dgf.base.encoder
report['encoder_children']={k:str(type(v)) for k,v in encoder.named_children()}
(HERE/'analysis.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
