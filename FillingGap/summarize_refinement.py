"""Post-training diagnostics. Frozen models only; no selection or exclusions."""
from pathlib import Path
import os,sys,json,hashlib,zipfile
os.environ.setdefault('MPLBACKEND','Agg')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from FillingGap.compare_mismatch_metrics import mismatch,summary
OUT=ROOT/'FillingGap/results_longrun_20261008'
def main():
 audit=json.loads((OUT/'completion_audit.json').read_text());assert audit['candidate_passes_numerical_gate']
 z=np.load(OUT/'final_test_baseline.npz');h=z['truth'].astype(float);pred=dict(np.load(OUT/'final_test_predictions.npz'))
 assert len(h)==1000
 psds=np.load(ROOT/'FillingGap/results_psd_reaudit/calibration_noise_psds.npz');sensitivity={}
 for key in ['boxcar_psd','hann_psd']:
  psd=psds[key].copy();psd[[0,-1]]=np.inf
  sensitivity[key]={name:summary(mismatch(h,p,psd)) for name,p in pred.items()}
  for reference in ['baseline','previous']:
   delta=mismatch(h,pred['longrun'],psd)-mismatch(h,pred[reference],psd)
   rng=np.random.default_rng(2026131002);ci=np.percentile(rng.choice(delta,size=(5000,len(delta)),replace=True).mean(1),[2.5,97.5])
   sensitivity[key]['vs_'+reference]={'improvement':1-sensitivity[key]['longrun']['mean']/sensitivity[key][reference]['mean'],'paired95ci':ci.tolist()}
  assert sensitivity[key]['vs_previous']['paired95ci'][1]<0
 psd=psds['psd'].copy();psd[[0,-1]]=np.inf
 transfer=np.load(ROOT/'FillingGap/results_model_optimization/linear_filter.npz')['transfer']
 linear=np.fft.irfft(np.fft.rfft(pred['baseline'],axis=1)*transfer,n=1056,axis=1)
 linear_scores=summary(mismatch(h,linear,psd))
 # Verify complete ablations from stored predictions, not only summary files.
 abdir=ROOT/'FillingGap/results_mask_frequency_ablation';ab=json.loads((abdir/'results.json').read_text());v=np.load(OUT/'validation_baseline.npz');vh=v['truth'].astype(float)
 peaks=np.argmax(abs(hilbert(vh,axis=1)),axis=1);pg=v['mask'][np.arange(len(vh)),peaks].astype(bool)
 assert len(ab['rows'])==12 and len({(r['variant'],r['seed']) for r in ab['rows']})==12
 for row in ab['rows']:
  p=np.load(abdir/f"{row['variant']}_{row['seed']}_validation_predictions.npz")['pred'];mm=mismatch(vh,p,psd);score=.75*mm.mean()+.25*mm[pg].mean();assert np.isclose(score,row['score'],rtol=1e-12)
 aggregate={}
 for name in ab['mean_score_by_variant']:
  rows=[r for r in ab['rows'] if r['variant']==name];aggregate[name]={'composite_score_mean':float(np.mean([r['score'] for r in rows])),'composite_score_sd':float(np.std([r['score'] for r in rows],ddof=1)),'population_mean':float(np.mean([r['population_mean'] for r in rows])),'peak_mean':float(np.mean([r['peak_mean'] for r in rows]))}
 wins={str(seed):min([r for r in ab['rows'] if r['seed']==seed],key=lambda r:r['score'])['variant'] for seed in sorted({r['seed'] for r in ab['rows']})}
 assert all(name=='mask_frequency' for name in wins.values())
 # Physical waveform/noise pairs must also be distinct between dataset roles.
 fingerprints={}
 for name in ['train','validation','final_test']:
  a=np.load(OUT/(name+'_raw.npz'));fingerprints[name]={hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest() for x,y in zip(a['truth'],a['noise'])};assert len(fingerprints[name])==len(a['truth'])
 assert not fingerprints['train']&fingerprints['validation'] and not fingerprints['train']&fingerprints['final_test'] and not fingerprints['validation']&fingerprints['final_test']
 result={'PSD_sensitivity':sensitivity,'frozen_linear_filter_baseline':linear_scores,'ablation_validation_only':aggregate,'ablation_winner_each_seed':wins,'physical_pair_no_cross_split_duplicates':True,'all_test_samples_retained':1000,'note':'Ablations use 2048 unique train signals and up to100 epochs, long run uses12000; three ablation seeds do not imply three full long-run replications.'}
 (OUT/'supplementary_checks.json').write_text(json.dumps(result,indent=2))
 peak=np.argmax(abs(hilbert(h,axis=1)),axis=1);peakgap=z['mask'][np.arange(len(h)),peak].astype(bool)
 fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
 for ax,keep,title in zip(axes,[np.ones(len(h),bool),peakgap],['All 1000 independent samples','Peak inside gap: 118 samples']):
  for name,label in [('baseline','DGF14'),('previous','Previous short refiner'),('longrun','Long-trained mask refiner')]:
   m=np.sort(mismatch(h,pred[name],psd)[keep]);ax.plot(np.maximum(m,1e-8),np.arange(1,len(m)+1)/len(m),label=label)
  ax.set_xscale('log');ax.set_xlabel('Calibrated PSD mismatch');ax.set_title(title);ax.grid(alpha=.25)
 axes[0].set_ylabel('CDF');axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'independent_test_cdf.png',dpi=200);fig.savefig(OUT/'independent_test_cdf.pdf');plt.close(fig)
 # Snapshot only imported project sources, preserving uncommitted dependencies.
 files={Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None) and Path(m.__file__).resolve().is_relative_to(ROOT) and Path(m.__file__).is_file() and str(m.__file__).endswith('.py')}
 files.update(p.resolve() for directory in ['data','utils','config'] for p in (ROOT/directory).rglob('*.py') if '__pycache__' not in p.parts)
 files.update(ROOT/'FillingGap'/name for name in ['train_longrun.py','audit_longrun.py','ablate_mask_frequency.py','infer_refined.py','summarize_refinement.py'])
 manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
 with zipfile.ZipFile(OUT/'runtime_sources.zip','w',zipfile.ZIP_DEFLATED) as archive:
  for p in sorted(files):archive.write(p,str(p.relative_to(ROOT)))
 (OUT/'runtime_source_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
