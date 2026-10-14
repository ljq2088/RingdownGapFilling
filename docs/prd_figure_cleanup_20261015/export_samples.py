from pathlib import Path
import json,sys,numpy as np
ROOT=Path('/home/ljq/code/RingdownGapFilling_PRD');sys.path.insert(0,str(ROOT))
from prd_conditioned.settings import OUT,CAL
from prd_physical.common import Preprocess,metric,sha
from prd_variable.data import variable_mask
HERE=Path(__file__).resolve().parent
rows=json.loads((HERE/'candidate_metrics.json').read_text());lookup={r['index']:r for r in rows}
chosen=[5946,1952,2314]
selection=[lookup[i] for i in chosen]
stats=np.load(OUT/'per_example_test.npz');mm=stats['variable__optimized']
diagnostic=np.argsort(mm,kind='stable')[[2999,5399,5999]]
seeds=np.load(OUT/'fresh_test/seeds.npy');mask=variable_mask(np.load(OUT/'fresh_test/mask.npy'),seeds,410000000)
pre=Preprocess(np.load(CAL)['psd'],'white')
for name,idx in [('figure4_native_samples',chosen),('diagnostic_native_samples',diagnostic)]:
 h=np.load(OUT/'fresh_test/truth.npy',mmap_mode='r')[idx].astype(float)
 noise=np.load(OUT/'fresh_test/noise.npy',mmap_mode='r')[idx].astype(float)
 p=pre.decode(np.load(OUT/'variable_optimized_prediction.npy',mmap_mode='r')[idx])
 m=mask[idx];g=h*(1-m[:,None]);obs=g+noise
 calculated=metric(h,p,np.load(CAL)['psd']);assert np.max(abs(calculated-mm[idx]))<1e-12
 np.savez_compressed(HERE/(name+'.npz'),time=np.arange(1056)/2,index=idx,seeds=seeds[idx],truth=h,noise=noise,mask=m,gapped_truth=g,observation=obs,reconstruction=p,mismatch=calculated)
audit=dict(checkpoint_sha256=sha(OUT/'finetune/best.pth'),selected=selection,
 diagnostic=[dict(index=int(i),seed=int(seeds[i]),mismatch=float(mm[i])) for i in diagnostic],
 selection_conditions=dict(peak_inside_gap=True,post_peak_damping_times_at_least=4,minimum_channel_energy_ratio=.1,minimum_visible_extrema=6,global_and_gap_error_below=.09,channel_peak_phase_degrees_below=1.5,native_curvature_below=.025),
 source_data_sha256={str(x.relative_to(OUT)):sha(x) for x in [OUT/'fresh_test/truth.npy',OUT/'fresh_test/noise.npy',OUT/'variable_optimized_prediction.npy']},
 population_statistics_unchanged=True,smoothing=False,resampling=False,time_alignment=False,phase_alignment=False)
assert all(r['relative_error']<.09 and r['gap_relative_error']<.09 and max(r['phase_A'],r['phase_E'])<1.5 and r['native_curvature']<.025 for r in selection)
(HERE/'figure_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
