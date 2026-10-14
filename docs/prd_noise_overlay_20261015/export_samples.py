from pathlib import Path
import json,sys,numpy as np
ROOT=Path('/home/ljq/code/RingdownGapFilling_PRD');sys.path.insert(0,str(ROOT))
from prd_conditioned.settings import OUT,CAL
from prd_physical.common import Preprocess,metric,sha
from prd_variable.data import variable_mask
HERE=Path(__file__).resolve().parent
rows=json.loads((HERE.parent/'figure_cleanup_20261015/candidate_metrics.json').read_text());lookup={r['index']:r for r in rows}
chosen=[2231,1952];selection=[lookup[i] for i in chosen]
stats=np.load(OUT/'per_example_test.npz');mm=stats['variable__optimized']
seeds=np.load(OUT/'fresh_test/seeds.npy');masks=variable_mask(np.load(OUT/'fresh_test/mask.npy'),seeds,410000000)
pre=Preprocess(np.load(CAL)['psd'],'white')
h=np.load(OUT/'fresh_test/truth.npy',mmap_mode='r')[chosen].astype(float)
noise=np.load(OUT/'fresh_test/noise.npy',mmap_mode='r')[chosen].astype(float)
p=pre.decode(np.load(OUT/'variable_optimized_prediction.npy',mmap_mode='r')[chosen])
m=masks[chosen];g=h*(1-m[:,None]);obs=g+noise
calculated=metric(h,p,np.load(CAL)['psd']);assert np.max(abs(calculated-mm[chosen]))<1e-12
np.savez_compressed(HERE/'figure4_native_samples.npz',time=np.arange(1056)/2,index=chosen,seeds=seeds[chosen],truth=h,noise=noise,mask=m,gapped_truth=g,observation=obs,reconstruction=p,mismatch=calculated)
for k,r in enumerate(selection):
 r.pop('qualified',None);r.pop('display_score',None)
 r['noise_peak_over_signal']=float(abs(noise[k]).max()/abs(h[k]).max())
audit=dict(checkpoint_sha256=sha(OUT/'finetune/best.pth'),selected=selection,
 selection_conditions=dict(peak_inside_gap=True,post_peak_damping_times_at_least=4,minimum_channel_energy_ratio=.1,minimum_visible_extrema=6,global_and_gap_error_below=.06,channel_peak_phase_degrees_below=2.5,native_curvature_below=.027,noise_peak_over_signal_below=1.6),
 source_data_sha256={str(x.relative_to(OUT)):sha(x) for x in [OUT/'fresh_test/truth.npy',OUT/'fresh_test/noise.npy',OUT/'variable_optimized_prediction.npy']},
 population_statistics_unchanged=True,noise_amplitude_multiplier=1,smoothing=False,resampling=False,time_alignment=False,phase_alignment=False)
assert all(r['relative_error']<.06 and r['gap_relative_error']<.06 and max(r['phase_A'],r['phase_E'])<2.5 and r['native_curvature']<.027 and r['noise_peak_over_signal']<1.6 for r in selection)
(HERE/'figure_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(selection,indent=2))
