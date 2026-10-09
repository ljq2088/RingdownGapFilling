"""Recompute manuscript diagnostics from frozen predictions, without training.

All test rows are retained. Input arrays use legacy benchmark amplitude units,
not a calibrated LISA strain or TDI observable. No time/phase maximization.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
from scipy.signal import hilbert, get_window, periodogram
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def stats(a):
    if not np.isfinite(a).all(): raise ValueError('nonfinite metric')
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p90=float(np.quantile(a,.9)),p95=float(np.quantile(a,.95)),max=float(a.max()))

def overlap(h,p,w=None,window=None):
    h=np.asarray(h,dtype=np.float64);p=np.asarray(p,dtype=np.float64)
    if window is not None: h=h*window;p=p*window
    if w is None: cross=(h*p).sum(1);hh=(h*h).sum(1);pp=(p*p).sum(1)
    else:
        H=np.fft.rfft(h,axis=1);P=np.fft.rfft(p,axis=1)
        cross=((H*P.conj()).real*w).sum(1)
        hh=(abs(H)**2*w).sum(1);pp=(abs(P)**2*w).sum(1)
    if np.any(hh<=0) or np.any(pp<=0): raise ValueError('zero waveform norm')
    return 1-np.clip(cross/np.sqrt(hh*pp),-1,1)

def main():
    a=argparse.ArgumentParser();a.add_argument('--input-root',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    root=args.input_root;out=args.output;out.mkdir(parents=True,exist_ok=True)
    src=root/'FillingGap/results_augmented_longrun_20261008'
    paths=[src/x for x in ['final_test_baseline.npz','final_test_raw.npz','final_test_predictions.npz','best_refiner.pth','config.json','history.json','final_results.json']]
    psdpath=root/'FillingGap/results_psd_reaudit/calibration_noise_psds.npz';paths.append(psdpath)
    with np.load(paths[0]) as z: data={k:z[k] for k in z.files}
    with np.load(paths[1]) as z: raw={k:z[k] for k in z.files}
    with np.load(paths[2]) as z: pred={k:z[k].astype(float) for k in z.files}
    with np.load(psdpath) as z: psds={k:z[k] for k in z.files}
    h=data['truth'].astype(float);m=data['mask'];n=raw['noise'].astype(float)
    assert np.array_equal(raw['truth'],data['truth'])
    assert np.array_equal(data['observation'],raw['truth']*(1-m)+raw['noise'])
    assert np.array_equal(data['baseline'],pred['baseline'])
    assert len(h)==1000 and h.shape[1]==1056
    assert np.array_equal(raw['seeds'],np.arange(2026240000,2026241000))
    assert np.all(m.sum(1)==66) and np.all((m==0)|(m==1))
    assert np.all((data['gap_starts']>=132)&(data['gap_starts']<=263))
    assert np.allclose(psds['frequency_hz'],np.fft.rfftfreq(1056,.5))
    for metric in ['psd','hann_psd']: assert np.all(np.isfinite(psds[metric])&(psds[metric]>0))
    # The seed ranges were frozen in training; no outcome filtering or selection.
    seeds=raw['seeds'];assert not np.isin(seeds,np.arange(2026101000,2026113000)).any()
    assert not np.isin(seeds,np.arange(2026121000,2026122500)).any()
    ah=hilbert(h,axis=1);peak=np.argmax(abs(ah),axis=1)
    pg=m[np.arange(len(h)),peak].astype(bool)
    r=np.linalg.norm(h,axis=1)/np.linalg.norm(n,axis=1)
    groups={'all':np.ones(len(h),bool),'peak_in_gap':pg,'peak_outside_gap':~pg}
    pnames=['baseline','previous','augmented']
    interp=data['observation'].astype(float).copy();t=np.arange(h.shape[1])
    for i in range(len(h)): interp[i,m[i].astype(bool)]=np.interp(t[m[i].astype(bool)],t[~m[i].astype(bool)],interp[i,~m[i].astype(bool)])
    pred['interpolation']=interp;pred['observation']=data['observation'].astype(float)
    weights={k:1/psds[k] for k in ['psd','hann_psd']}
    for w in weights.values(): w[[0,-1]]=0;w/=w.mean()
    hann=get_window('hann',h.shape[1]);metrics={};per={};rng=np.random.default_rng(20261009)
    for name,p in pred.items():
        assert p.shape==h.shape and np.isfinite(p).all()
        metrics[name]={
            'time_mismatch':overlap(h,p),
            'psd_mismatch':overlap(h,p,weights['psd']),
            'hann_psd_mismatch':overlap(h,p,weights['hann_psd'],hann),
            'relative_l2':np.linalg.norm(p-h,axis=1)/np.linalg.norm(h,axis=1),
            'gap_relative_l2':np.sqrt(((p-h)**2*m).sum(1)/(h*h*m).sum(1)),
            'gap_error_global_norm':np.sqrt(((p-h)**2*m).sum(1)/(h*h).sum(1)),
            'peak_abs_phase_deg':abs(np.angle(hilbert(p,axis=1)[np.arange(len(h)),peak]*ah[np.arange(len(h)),peak].conj()))*180/np.pi,
            'signed_norm_bias':np.linalg.norm(p,axis=1)/np.linalg.norm(h,axis=1)-1,
        }
        per.update({name+'__'+k:v for k,v in metrics[name].items()})
    report=dict(contract='legacy synthetic benchmark; d=G*h+n; all rows retained',
                masked_energy_fraction=stats((h*h*m).sum(1)/(h*h).sum(1)),
                sample_count=len(h),peak_in_gap_count=int(pg.sum()),
                R=stats(r),groups={g:{name:{k:stats(v[keep]) for k,v in mm.items()} for name,mm in metrics.items()} for g,keep in groups.items()},
                paired={},source_sha256={str(p.relative_to(root)):sha(p) for p in paths},
                selected_epoch=json.loads((src/'final_results.json').read_text())['selected_epoch'],
                selection_rule='0.75*validation population mean + 0.25*validation peak-in-gap mean',
                confidence_scope='paired sample bootstrap, fixed fitted checkpoints; not training-seed uncertainty')
    for g,keep in groups.items():
        report['paired'][g]={}
        for ref in ['baseline','previous','interpolation']:
            d=metrics['augmented']['psd_mismatch'][keep]-metrics[ref]['psd_mismatch'][keep]
            bs=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(5000)])
            report['paired'][g][ref]=dict(mean_delta=float(d.mean()),ci95=np.quantile(bs,[.025,.975]).tolist())
    # Independent recomputation must reproduce the trainer's final table.
    expected=json.loads((src/'final_results.json').read_text())['rows'][0]['augmented']['mean']
    assert abs(report['groups']['all']['augmented']['psd_mismatch']['mean']-expected)<1e-12
    np.savez_compressed(out/'per_example_metrics.npz',seeds=seeds,peak_in_gap=pg,R=r,**per)
    (out/'evaluation.json').write_text(json.dumps(report,indent=2))
    plt.rcParams.update({'font.size':9,'axes.labelsize':9,'legend.fontsize':8,'pdf.fonttype':42})
    colors={'baseline':'#777777','previous':'#0072B2','augmented':'#D55E00','interpolation':'#009E73'}
    labels={'baseline':'Original DGF','previous':'First refinement','augmented':'Augmented refinement','interpolation':'Linear interpolation'}
    fig,ax=plt.subplots(1,2,figsize=(7.1,2.7),constrained_layout=True)
    for name in colors:
        for j,k in enumerate(['time_mismatch','psd_mismatch']):
            x=np.sort(metrics[name][k]);ax[j].plot(x,np.arange(1,len(x)+1)/len(x),color=colors[name],label=labels[name])
    for j in range(2): ax[j].set_xscale('log');ax[j].set_xlim(1e-5,2);ax[j].set_ylim(0,1);ax[j].set_ylabel('Fraction of test examples');ax[j].grid(alpha=.2)
    ax[0].set_xlabel('Direct time-domain mismatch');ax[1].set_xlabel('Direct PSD-weighted mismatch');ax[0].legend(loc='lower right')
    fig.savefig(out/'mismatch_cdf.pdf');fig.savefig(out/'mismatch_cdf.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(7.1,2.7),constrained_layout=True)
    for name in pnames:
        x=np.sort(metrics[name]['peak_abs_phase_deg']);ax[0].plot(x,np.arange(1,1001)/1000,color=colors[name],label=labels[name])
        ax[1].scatter(r,metrics[name]['psd_mismatch'],s=5,alpha=.3,color=colors[name])
    ax[0].set_xlabel('Absolute phase deviation at target envelope peak (deg)');ax[0].set_ylabel('Fraction of test examples');ax[0].set_xscale('symlog',linthresh=.2);ax[0].set_xlim(0,180);ax[0].set_ylim(0,1);ax[0].legend(loc='lower right')
    ax[1].set_xscale('log');ax[1].set_yscale('log');ax[1].set_xlabel(r'$R=\|h\|_2/\|n\|_2$');ax[1].set_ylabel('Direct PSD-weighted mismatch')
    fig.savefig(out/'phase_and_R.pdf');plt.close(fig)
    # Describe central, upper-tail and worst cases; show the same fixed data for both models.
    ordered=np.argsort(metrics['augmented']['psd_mismatch']);indices=[int(ordered[499]),int(ordered[899]),int(ordered[-1])]
    report['example_indices']=indices;report['example_seeds']=[int(seeds[i]) for i in indices]
    fig,axes=plt.subplots(3,1,figsize=(7.1,5.3),constrained_layout=True)
    for ax,i in zip(axes,indices):
        ax.plot(t/2,h[i]*1e22,color='k',lw=1,label='Clean target')
        for name in ['baseline','augmented']: ax.plot(t/2,pred[name][i]*1e22,lw=.9,color=colors[name],label=labels[name])
        st=data['gap_starts'][i];ax.axvspan(st/2,(st+66)/2,color='gray',alpha=.22)
        ax.set_title('Test seed '+str(seeds[i])+'; mismatch = '+format(metrics['augmented']['psd_mismatch'][i],'.4g'),fontsize=9)
        ax.set_xlim(max(0,(st-110)/2),min(528,(st+180)/2));ax.set_ylabel(r'Benchmark amplitude $\times10^{22}$')
    axes[0].legend(loc='upper right');axes[-1].set_xlabel('Time from segment start (s)')
    fig.savefig(out/'examples.pdf');plt.close(fig)
    history=json.loads((src/'history.json').read_text());fig,ax=plt.subplots(figsize=(3.45,2.5),constrained_layout=True)
    ax.plot([x['epoch'] for x in history],[x['population_mean'] for x in history],label='Population mean')
    ax.plot([x['epoch'] for x in history],[x['validation_score'] for x in history],label='Selection score')
    ax.axvline(1095,color='k',ls='--',lw=.8,label='Selected epoch');ax.set_xlabel('Refinement epoch');ax.set_ylabel('Validation mismatch / score');ax.legend()
    fig.savefig(out/'training.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(3.45,2.5),constrained_layout=True)
    f,pn=periodogram(n,fs=2,axis=1)
    ax.loglog(f[1:-1],pn.mean(0)[1:-1],color='k',label='Simulated noise')
    for name in ['baseline','augmented']:
        _,pe=periodogram(pred[name]-h,fs=2,axis=1);ax.loglog(f[1:-1],pe.mean(0)[1:-1],color=colors[name],label=labels[name]+' error')
    ax.set_xlabel('Frequency (Hz)');ax.set_ylabel('PSD (benchmark units squared / Hz)');ax.legend(fontsize=7)
    fig.savefig(out/'error_psd.pdf');plt.close(fig)
    (out/'evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({g:report['groups'][g]['augmented'] for g in ['all','peak_in_gap']},indent=2))

if __name__=='__main__': main()
