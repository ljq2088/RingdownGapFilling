from pathlib import Path
import sys,json,numpy as np
from scipy.signal import hilbert
ROOT=Path('/home/ljq/code/RingdownGapFilling_PRD');sys.path.insert(0,str(ROOT))
from prd_tail.settings import OUT,CACHE,CAL
from prd_physical.common import Preprocess,summary
from pycbc.conversions import get_lm_f0tau
HERE=Path(__file__).resolve().parent
pre=Preprocess(np.load(CAL)['psd'],'white');p=pre.decode(np.load(OUT/'cache/validation0/prediction.npy'))
h=np.load(CACHE/'validation_truth.npy').astype(float);pars=np.load(CACHE/'validation_parameters.npy');peak=np.load(OUT/'cache/validation0/peak.npy')
indices=np.random.default_rng(2026101561).choice(len(h),128,replace=False);time=np.arange(1056)*.5;rows=[]
for i in indices:
    anchor=min(496.,peak[i]+64);u=np.maximum(time-anchor,0)
    f,tau=np.array([get_lm_f0tau(pars[i,9],pars[i,10],l,m) for l,m in [(2,2),(2,1),(3,3),(4,4)]]).T
    D=np.stack((np.exp(-u[:,None]/tau)*np.cos(2*np.pi*u[:,None]*f),np.exp(-u[:,None]/tau)*np.sin(2*np.pi*u[:,None]*f)),-1).reshape(1056,8)
    region=time>=anchor;fit=region&(time<=anchor+min(3*tau[0],220));evaluation=region&(~fit)
    if fit.sum()<32 or evaluation.sum()<32:continue
    scale=np.max(abs(h[i]));truth=h[i]/scale;pred=p[i]/scale
    # This is a capacity oracle using known validation source parameters and
    # clean fit samples. It is never an inference result or reported benchmark.
    coef=np.linalg.lstsq(D[fit],truth[:,fit].T,rcond=1e-8)[0]
    oracle=(D@coef).T
    denominator=(truth[:,evaluation]**2).sum()
    if denominator<1e-16:continue
    rows.append(dict(index=int(i),known_parameter_oracle_relative_error=float(np.sqrt(((oracle-truth)[:,evaluation]**2).sum()/denominator)),baseline_relative_error=float(np.sqrt(((pred-truth)[:,evaluation]**2).sum()/denominator))))
report=dict(diagnostic_only=True,uses_known_source_parameters_and_clean_fit_samples=True,never_used_as_reconstruction_predictions=True,validation_sources_probed=128,eligible_probe_count=len(rows),oracle=summary([r['known_parameter_oracle_relative_error'] for r in rows]),baseline=summary([r['baseline_relative_error'] for r in rows]),rows=rows)
(HERE/'qnm_capacity.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
