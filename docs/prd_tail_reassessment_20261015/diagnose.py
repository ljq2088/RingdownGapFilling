from pathlib import Path
import sys,json,numpy as np
from scipy.signal import hilbert
ROOT=Path('/home/ljq/code/RingdownGapFilling_PRD');sys.path.insert(0,str(ROOT))
from prd_conditioned.settings import CACHE,CAL,START as PREVIOUS
from prd_physical.common import Preprocess,metric,summary
HERE=Path(__file__).resolve().parent
pre=Preprocess(np.load(CAL)['psd'],'white')
truth=np.load(CACHE/'validation_truth.npy').astype(float)
white=np.load(HERE.parent/'optimization_round_20261015/validation_baseline_white.npy')
pred=pre.decode(white)
f=np.fft.rfft(pred);ends=np.zeros_like(f);ends[...,[0,-1]]=f[...,[0,-1]];endpoint=np.fft.irfft(ends,n=1056)
t=np.arange(1056)/2
z=hilbert(truth);w=hilbert(pred);amp=abs(z);peak=amp.argmax(-1);hmax=amp.max(-1,keepdims=True)
band={}
for name,lo,hi in [('strong',.1,np.inf),('intermediate',.01,.1),('weak',.001,.01)]:
    selected=(amp>=lo*hmax)&(amp<hi*hmax)&(np.arange(1056)[None,None,:]>=peak[...,None])
    den=(truth**2*selected).sum((1,2));valid=den>0
    err=np.sqrt(((pred-truth)**2*selected).sum((1,2))[valid]/den[valid])
    local=((np.angle(w*z.conj()))**2*amp**2*selected).sum((1,2))/(amp**2*selected).sum((1,2)).clip(1e-100)
    ep=((endpoint**2)*selected).sum((1,2))/den.clip(1e-100)
    band[name]={'sources_with_region':int(valid.sum()),'relative_error':summary(err),
                'envelope_weighted_phase_rms_degrees':summary(np.sqrt(local[valid])*180/np.pi),
                'endpoint_energy_over_target_energy':summary(ep[valid])}
report={'validation_sources':len(truth),'baseline_direct_mismatch':summary(metric(truth,pred,np.load(CAL)['psd'])),
        'after_removing_predicted_dc_nyquist_direct_mismatch':summary(metric(truth,pred-endpoint,np.load(CAL)['psd'])),
        'regions':band,'diagnostic_only_true_envelope_used_to_define_regions':True}
(HERE/'diagnosis.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
