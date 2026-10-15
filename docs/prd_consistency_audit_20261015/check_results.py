from pathlib import Path
import json, hashlib
import numpy as np
from scipy.signal import hilbert, periodogram

P=Path(__file__).resolve().parent
R=Path(r'\\wsl.localhost\Ubuntu-22.04-D\home\ljq\code\RingdownGapFilling_PRD')
F=R/'FillingGap/results_prd_tail_20261015'
psd=np.load(R/'FillingGap/results_prd_physical_transfer_20261009/calibration_psd.npz')['psd']
asd=np.sqrt(psd);asd[:,0]=asd[:,1];asd[:,-1]=asd[:,-2]
hbank=np.load(F/'fresh_test/truth.npy',mmap_mode='r')
nbank=np.load(F/'fresh_test/noise.npy',mmap_mode='r')
original=np.load(F/'fresh_test/mask.npy',mmap_mode='r')
seeds=np.load(F/'fresh_test/seeds.npy')
stored=np.load(F/'per_example_test.npz')
pub=np.load(P.parent/'new_model_manuscript_20261015/publication_metrics.npz')
report={}
def summary(x):
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),p95=float(np.quantile(x,.95)),min=float(x.min()),max=float(x.max()))
for family in ['variable','fixed66','fixed100']:
    pred=np.load(F/(family+'_optimized_prediction.npy'),mmap_mode='r')
    vals={k:[] for k in ['snr','mismatch','time_mm','phase_peak','phase_gap','gap_relative','energy','ES','inside']}
    spectra={k:np.zeros((2,529)) for k in ['input','residual','noise']}
    errors=[]
    for lo in range(0,6000,256):
        hi=min(lo+256,6000);h=hbank[lo:hi].astype(float);n=nbank[lo:hi].astype(float)
        p=np.fft.irfft(np.fft.rfft(pred[lo:hi].astype(float))*asd,n=1056)
        lengths=np.array([64+2*int(np.random.default_rng(int(s)+410000000).integers(21)) for s in seeds[lo:hi]]) if family=='variable' else np.full(hi-lo,int(family[5:]))
        starts=original[lo:hi].argmax(-1)-(lengths-66)//2
        m=(np.arange(1056)[None,:]>=starts[:,None])&(np.arange(1056)[None,:]<(starts+lengths)[:,None])
        # Physical Fourier map for the stated one-sided inner product.
        A=np.fft.rfft(h)[...,1:-1]*.5/np.sqrt(psd[:,1:-1])
        B=np.fft.rfft(p)[...,1:-1]*.5/np.sqrt(psd[:,1:-1])
        hh=(abs(A)**2).sum((1,2));pp=(abs(B)**2).sum((1,2))
        mm=1-np.clip((A*B.conj()).real.sum((1,2))/np.sqrt(hh*pp),-1,1)
        rho=np.sqrt(4*(2/1056)*hh)
        tm=1-np.clip((h*p).sum((1,2))/np.sqrt((h*h).sum((1,2))*(p*p).sum((1,2))),-1,1)
        z=hilbert(h);w=hilbert(p);amp2=abs(z)**2;arg=np.angle(w*z.conj())
        ix=amp2.argmax(-1);peakphase=abs(np.take_along_axis(arg,ix[...,None],axis=-1)[...,0])*180/np.pi
        joint=amp2.sum(1).argmax(-1);inside=m[np.arange(hi-lo),joint]
        gaprel=np.sqrt((((p-h)**2)*m[:,None]).sum((1,2))/((h*h)*m[:,None]).sum((1,2)))
        gap_phase=np.sqrt((amp2*arg**2*m[:,None]).sum((1,2))/(amp2*m[:,None]).sum((1,2)))*180/np.pi
        es=(abs(B-A)**2).sum((1,2))/hh
        energy=((h*h)*m[:,None]).sum((1,2))/(h*h).sum((1,2))
        for k,v in dict(snr=rho,mismatch=mm,time_mm=tm,phase_peak=peakphase,phase_gap=gap_phase,gap_relative=gaprel,energy=energy,ES=es,inside=inside).items(): vals[k].append(v)
        errors.append(float(np.max(abs(stored[family+'__optimized'][lo:hi]-mm))))
        if family=='variable':
            # Parseval equality of the declared metric and whitened norm.
            white=np.fft.irfft(np.fft.rfft(h)/asd,n=1056)
            ft=np.fft.rfft(white);ft[...,0]=0;ft[...,-1]=0
            assert np.allclose((np.fft.irfft(ft,n=1056)**2).sum((1,2)),rho**2,rtol=1e-12)
            for k,x in [('input',n-m[:,None]*h),('residual',p-h),('noise',n)]:
                ff,s=periodogram(x,fs=2,window='boxcar',detrend='constant',scaling='density',axis=-1)
                spectra[k]+=s.sum(0)/6000
    vals={k:np.concatenate(v) for k,v in vals.items()};sn=vals['snr']>=10
    out={'all':{k:summary(vals[k]) for k in ['mismatch','time_mm','ES']},'snr_ge10':{k:summary(vals[k][sn]) for k in ['mismatch','time_mm','ES']},'phase_peak':{c:summary(vals['phase_peak'][:,j]) for j,c in enumerate('AE')},'stored_mismatch_max_abs_difference':max(errors)}
    if family=='variable':
        for key in ['snr','mismatch','time_mm','phase_peak','phase_gap','gap_relative','energy','ES','inside']:
            assert np.allclose(vals[key],pub[key],rtol=1e-10,atol=1e-12),key
        inside=vals['inside'];out['snr']=summary(vals['snr'])
        out['inside']={k:summary(vals[k][inside]) for k in ['mismatch','phase_gap','gap_relative']}
        out['inside_snr_ge10']={k:summary(vals[k][inside&sn]) for k in ['mismatch','phase_gap','gap_relative']}
        out['outside_mismatch']=summary(vals['mismatch'][~inside])
        out['energy_snr_ge10']=summary(vals['energy'][sn])
        out['low_energy_mismatch']=summary(vals['mismatch'][sn&(vals['energy']<.25)])
        out['high_energy_mismatch']=summary(vals['mismatch'][sn&(vals['energy']>=.75)])
        out['all_interior_psd_bins_improved']=bool(np.all(spectra['residual'][:,1:-1]<spectra['input'][:,1:-1]))
        np.savez_compressed(P/'recomputed_metrics.npz',**vals)
        np.savez_compressed(P/'recomputed_spectra.npz',frequency=ff,**spectra)
    report[family]=out
manifest=json.loads((P/'source/reproducibility_manifest.json').read_text())
report['figure_hashes']={k:hashlib.sha256((P/'source'/k).read_bytes()).hexdigest()==v for k,v in manifest['figure_sha256'].items()}
assert all(report['figure_hashes'].values())
assert max(report[f]['stored_mismatch_max_abs_difference'] for f in ['variable','fixed66','fixed100'])<1e-12
(P/'numerical_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
