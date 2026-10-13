"""Post-test SNR-conditioned diagnostics; frozen models and full-population result retained."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'2')
import json
from pathlib import Path
import numpy as np
from prd_physical.common import N,FS,write,summary
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'FillingGap/results_prd_lowrank_20261013'
OUT=ROOT/'FillingGap/results_prd_snr_conditioning_20261013'
CAL=ROOT/'FillingGap/results_prd_physical_transfer_20261009/calibration_psd.npz'
THRESHOLDS=(0.,5.,8.,10.,15.,20.,30.,50.)
MODELS=('previous_joint','frequency_snapshot','fixed_projection','repeated_control','noise_augmented')
SEED=2026101371

def snr(signal,psd):
    h=np.asarray(signal,np.float64)
    f=np.fft.rfft(h,axis=-1)
    value=(4/(FS*N)*(abs(f[...,1:-1])**2/psd[:,1:-1])).sum((1,2))
    # Independent real-sample Parseval check with the actual whitening convention.
    work=f/np.sqrt(psd*FS/2)
    work[...,0]=work[...,-1]=0
    white=np.fft.irfft(work,n=N,axis=-1)
    check=(white*white).sum((1,2))
    error=float(np.max(abs(check-value)/np.maximum(value,1e-300)))
    assert error<1e-12 and np.isfinite(value).all() and (value>0).all()
    return np.sqrt(value),error

def conditioning(rho,per):
    rows=[];rng=np.random.default_rng(SEED);rng_mean=np.random.default_rng(SEED+567)
    for threshold in THRESHOLDS:
        kept=rho>=threshold;removed=~kept;n=int(kept.sum())
        if n==0:continue
        metrics={}
        for model in MODELS:
            metrics[model]={name:summary(per[model+'__'+name][kept]) for name in ('psd','time','hann_psd','relative_l2')}
        delta=per['noise_augmented__psd'][kept]-per['previous_joint__psd'][kept]
        boot=np.array([rng.choice(delta,n,replace=True).mean() for _ in range(2000)])
        current=per['noise_augmented__psd'];mean_boot=np.array([rng_mean.choice(current[kept],n,replace=True).mean() for _ in range(2000)]);ref=metrics['previous_joint']['psd']['mean'];score=metrics['noise_augmented']['psd']['mean']
        row=dict(snr_min=threshold,mean_mismatch_ci95=np.quantile(mean_boot,[.025,.975]).tolist(),retained=n,excluded=int(removed.sum()),retained_percent=100*n/len(rho),excluded_percent=100*removed.mean(),metrics=metrics,reduction_vs_same_subset_previous_percent=100*(ref-score)/ref,excluded_mismatch_sum_fraction=float(current[removed].sum()/current.sum()),excluded_mean_mismatch=float(current[removed].mean()) if removed.any() else None,paired_vs_previous=dict(mean_delta=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist()))
        rows.append(row)
    bins=[]
    edges=(0.,5.,8.,10.,15.,20.,30.,50.,np.inf)
    for lo,hi in zip(edges[:-1],edges[1:]):
        chosen=(rho>=lo)&(rho<hi)
        if chosen.any():bins.append(dict(low=lo,high=hi if np.isfinite(hi) else None,n=int(chosen.sum()),new=summary(per['noise_augmented__psd'][chosen]),previous=summary(per['previous_joint__psd'][chosen])))
    return dict(thresholds=rows,bins=bins,snr_summary=summary(rho))

def main():
    OUT.mkdir(exist_ok=True)
    final=json.loads((RUN/'final_results.json').read_text());assert final['all_rows_retained'] and final['n']==6000
    truth=np.load(RUN/'fresh_test/truth.npy',mmap_mode='r').astype(float)
    mask=np.load(RUN/'fresh_test/mask.npy',mmap_mode='r')
    psd=np.load(CAL)['psd']
    full,e1=snr(truth,psd);observed,e2=snr(truth*(1-mask[:,None]),psd)
    with np.load(RUN/'per_example_test.npz') as f:per={k:f[k] for k in f.files}
    assert np.array_equal(per['seeds'],np.load(RUN/'fresh_test/seeds.npy'))
    for model in MODELS:
        assert abs(float(per[model+'__psd'].mean())-final['results'][model]['psd']['mean'])<1e-15
    result=dict(passed=True,n_all=6000,primary_full_population_mean=final['results']['noise_augmented']['psd']['mean'],conditioning='post-test descriptive diagnostics; thresholds fixed before this calculation; predictions and model selection unchanged',full_signal_snr=conditioning(full,per),gated_signal_snr=conditioning(observed,per),snr_definition='rho^2=4/(fs*N) sum_AE sum_k=1..N/2-1 |DFT(signal)|^2/PSD; same fixed finite-crop PSD and endpoints as main metric',snr_parseval_relative_errors=dict(full=e1,gated=e2),rho_gated_over_full=summary(observed/full),bootstrap_scope='paired source bootstrap conditional on threshold and checkpoints; not training-seed uncertainty',test_seed_range=[2034200000,2034205999],full_population_result_retained=True,inference_selection_scope='clean h and G*h available in simulation; observed-data selection needs an independently specified SNR estimator')
    write(OUT/'snr_conditioned_results.json',result)
    np.savez_compressed(OUT/'snr_per_source.npz',seeds=per['seeds'],full_signal_snr=full,gated_signal_snr=observed)
    for definition in ['full_signal_snr','gated_signal_snr']:
        print(definition)
        for row in result[definition]['thresholds']:
            print(row['snr_min'],row['retained'],round(row['excluded_percent'],3),row['metrics']['noise_augmented']['psd']['mean'],row['metrics']['previous_joint']['psd']['mean'],round(row['reduction_vs_same_subset_previous_percent'],3),round(row['excluded_mismatch_sum_fraction'],3))
    print('SNRchecks',result['snr_parseval_relative_errors'])
if __name__=='__main__':main()
