"""Audit completed results; never selects models or changes training/test data."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.signal import hilbert
from FillingGap.compare_mismatch_metrics import mismatch,summary
OUT=ROOT/'FillingGap/results_longrun_20261008'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 status=json.loads((OUT/'status.json').read_text())
 if status['stage']!='completed':raise SystemExit('Not completed: audit must wait for the final test.')
 config=json.loads((OUT/'config.json').read_text());report=json.loads((OUT/'final_results.json').read_text())
 assert config['observation']=='G*h+n'
 assert sha(ROOT/'saved_models/noise/lowSNR/model_with_noise_gaps_HugeRangeParas_14.pth')==config['baseline_sha256']
 assert sha(ROOT/'FillingGap/results_model_optimization/best_refiner.pth')==config['warmstart_sha256']
 assert sha(ROOT/'FillingGap/results_psd_reaudit/calibration_noise_psds.npz')==config['psd_sha256']
 archives={k:np.load(OUT/(k+'_raw.npz')) for k in ['train','validation','final_test']}
 seeds={k:set(z['seeds'].tolist()) for k,z in archives.items()}
 assert not seeds['train']&seeds['validation'] and not seeds['train']&seeds['final_test'] and not seeds['validation']&seeds['final_test']
 assert [len(seeds[k]) for k in ['train','validation','final_test']]==[config['train_unique'],config['validation_natural'],config['final_test']]
 old=np.load(ROOT/'data/cqg_fresh_refiner_test_seed2026100800_n500.npz');prior_seeds=set(old['generation_seeds'].tolist())
 prior_overlap={k:len(prior_seeds&v) for k,v in seeds.items()}
 assert prior_overlap['validation']==0 and prior_overlap['final_test']==0
 # Check exact array construction in the float32 cache, not only a string in config.
 invariants={}
 for name in archives:
  z=np.load(OUT/(name+'_baseline.npz'));h,n=archives[name]['truth'],archives[name]['noise']
  if name=='train':h=np.concatenate([h,h]);n=np.concatenate([n,n])
  assert np.array_equal(h,z['truth']);assert np.isin(z['mask'],[0,1]).all()
  assert np.array_equal(z['observation'],h*(1-z['mask'])+n)
  assert np.all(z['mask'].sum(1)==66);assert np.all((z['gap_starts']>=132)&(z['gap_starts']<=263))
  invariants[name]={'unique_signals':len(seeds[name]),'masked_examples':len(h),'observation_exact_Gh_plus_n':True}
 z=np.load(OUT/'final_test_baseline.npz');pred=np.load(OUT/'final_test_predictions.npz');h=z['truth'].astype(float)
 for key in ['longrun','previous','baseline']:assert pred[key].shape==h.shape and np.isfinite(pred[key]).all()
 assert np.array_equal(pred['baseline'],z['baseline'].astype(float))
 psd=np.load(ROOT/'FillingGap/results_psd_reaudit/calibration_noise_psds.npz')['psd'].copy();psd[[0,-1]]=np.inf
 peak=np.argmax(abs(hilbert(h,axis=1)),axis=1);pg=z['mask'][np.arange(len(h)),peak].astype(bool)
 groups={'all':np.ones(len(h),bool),'peak_in_gap':pg};metrics={};rng=np.random.default_rng(2026131001)
 for group,keep in groups.items():
  metrics[group]={}
  truthphase=hilbert(h,axis=1)[np.arange(len(h)),peak]
  for key in ['baseline','previous','longrun']:
   p=pred[key];mm=mismatch(h,p,psd);phase=np.angle(hilbert(p,axis=1)[np.arange(len(h)),peak]*truthphase.conj())*180/np.pi
   metrics[group][key]={'n':int(keep.sum()),'PSD':summary(mm[keep]),'time':summary(mismatch(h,p)[keep]),'absolute_peak_phase_deg':{'mean':float(abs(phase[keep]).mean()),'median':float(np.median(abs(phase[keep])))}}
  for reference in ['baseline','previous']:
   delta=(mismatch(h,pred['longrun'],psd)-mismatch(h,pred[reference],psd))[keep]
   ci=np.percentile(rng.choice(delta,size=(5000,len(delta)),replace=True).mean(1),[2.5,97.5])
   metrics[group]['vs_'+reference]={'mean_improvement_fraction':1-metrics[group]['longrun']['PSD']['mean']/metrics[group][reference]['PSD']['mean'],'paired_mean_delta':float(delta.mean()),'paired_bootstrap95ci':ci.tolist()}
 # Confirm published test summaries agree with direct recomputation.
 for row in report['rows']:
  reference='baseline' if row['comparison']=='DGF14' else 'previous';group=row['group']
  assert row['n']==metrics[group]['longrun']['n']
  assert np.isclose(row['longrun']['mean'],metrics[group]['longrun']['PSD']['mean'],rtol=1e-12)
  assert np.isclose(row['reference']['mean'],metrics[group][reference]['PSD']['mean'],rtol=1e-12)
 gates={'mean_at_least_10pct_better_than_previous':metrics['all']['vs_previous']['mean_improvement_fraction']>=.10,'paired_CI_below_zero_vs_previous':metrics['all']['vs_previous']['paired_bootstrap95ci'][1]<0,'paired_CI_below_zero_vs_original':metrics['all']['vs_baseline']['paired_bootstrap95ci'][1]<0,'no_significant_peak_regression_vs_previous':metrics['peak_in_gap']['vs_previous']['paired_bootstrap95ci'][0]<=0}
 result={'data_and_hash_checks_passed':True,'invariants':invariants,'prior_fresh500_seed_overlap':prior_overlap,'prior_fresh500_independent_for_this_model':False,'metrics':metrics,'gates':gates,'candidate_passes_numerical_gate':all(gates.values()),'selected_epoch':report['selected_epoch'],'limitations':'same simulator/distribution; one training seed; prior fresh500 partly overlaps this training and must not serve as independent long-run evidence; ablation and seed-stability evidence remain separate requirements'}
 (OUT/'completion_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
