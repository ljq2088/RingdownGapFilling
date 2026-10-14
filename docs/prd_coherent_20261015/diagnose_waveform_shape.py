from pathlib import Path
import sys,json,numpy as np
ROOT=Path('/home/ljq/code/RingdownGapFilling_PRD');sys.path.insert(0,str(ROOT))
from prd_conditioned.settings import OUT,CAL
from prd_physical.physics import PARAM_NAMES,REF_SECONDS,FS
from prd_physical.prepare import crop_meta
from pycbc.conversions import get_lm_f0tau
from scipy.signal import hilbert,find_peaks
p=np.load(OUT/'fresh_test/parameters.npy');h=np.load(OUT/'fresh_test/truth.npy',mmap_mode='r');seeds=np.load(OUT/'fresh_test/seeds.npy');stats=np.load(OUT/'per_example_test.npz')
indices=[1116,1885,1777,1873,5865,2685];reports=[]
for i in indices:
    row={k:float(v) for k,v in zip(PARAM_NAMES,p[i])};start,m=crop_meta(p[i],int(seeds[i]));modes={}
    for l,k in [(2,2),(2,1),(3,3),(4,4)]:
        f,tau=get_lm_f0tau(p[i,9],p[i,10],l,k);modes[str(l)+str(k)]=dict(frequency_hz=float(f),period_seconds=float(1/f),tau_seconds=float(tau),cycles_in_528_seconds=float(528*f),cycles_per_damping_time=float(f*tau))
    curve=np.asarray(h[i],float);onset=REF_SECONDS-start/FS
    energy=np.sum(curve**2,axis=0);t=np.arange(1056)/2;inside=(t>=onset)&(t<=onset+100)
    reports.append(dict(index=i,seed=int(seeds[i]),parameters=row,crop_start_parent_seconds=start/FS,source_reference_local_seconds=onset,modes=modes,network_snr=float(stats['snr'][i]),mismatch=float(stats['variable__optimized'][i]),joint_time_energy_within_100s_of_source_onset=float(energy[inside].sum()/energy.sum()),channel_zero_crossings=[int(np.sum(np.diff(np.sign(x))!=0)) for x in curve]))
allf=np.array([get_lm_f0tau(x[9],x[10],2,2)[0] for x in p]);allt=np.array([get_lm_f0tau(x[9],x[10],2,2)[1] for x in p]);r=dict(cases=reports,population=dict(n=len(p),frequency_220_min=float(allf.min()),frequency_220_max=float(allf.max()),tau_220_min=float(allt.min()),tau_220_max=float(allt.max()),less_than_one_cycle_in_record=int((528*allf<1).sum()),less_than_two_cycles_in_record=int((528*allf<2).sum()),tau_exceeds_window=int((allt>528).sum())))
dest=Path(__file__).resolve().parent/'waveform_shape_diagnosis.json';dest.write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
