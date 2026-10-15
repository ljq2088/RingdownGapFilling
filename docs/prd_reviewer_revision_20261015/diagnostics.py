import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
from pathlib import Path
import json
import numpy as np
from scipy.signal import hilbert,periodogram
from scipy.linalg import cho_factor,cho_solve
from pycbc.conversions import get_lm_f0tau

P=Path(__file__).resolve().parent;R=Path('/home/ljq/code/RingdownGapFilling_PRD')
F=R/'FillingGap/results_prd_conditioned_20261014';T=F/'fresh_test';N=1056;fs=2.
h=np.load(T/'truth.npy',mmap_mode='r');noise=np.load(T/'noise.npy',mmap_mode='r');prediction=np.load(F/'variable_optimized_prediction.npy',mmap_mode='r')
psd=np.load(R/'FillingGap/results_prd_physical_transfer_20261009/calibration_psd.npz')['psd'].astype(float)
asd=np.sqrt(psd*fs/2);asd[:,0]=asd[:,1];asd[:,-1]=asd[:,-2]
seeds=np.load(T/'seeds.npy');base=np.load(T/'mask.npy',mmap_mode='r');snr=np.load(F/'per_example_test.npz')['snr'];strong=snr>=10
length=np.array([64+2*int(np.random.default_rng(int(s)+410000000).integers(21)) for s in seeds]);start=base.argmax(-1)-(length-66)//2
mask=((np.arange(N)[None,:]>=start[:,None])&(np.arange(N)[None,:]<start[:,None]+length[:,None]))
inside=np.empty(len(h),bool);phase_gap=np.empty(len(h));phase_peak=np.empty((len(h),2));energy=np.empty(len(h));rel=np.empty(len(h));ES=np.empty(len(h));sums={k:np.zeros((2,N//2+1)) for k in ('target','noise','residual')}
for lo in range(0,len(h),128):
    hi=min(lo+128,len(h));truth=np.asarray(h[lo:hi],float);n=np.asarray(noise[lo:hi],float)
    p=np.fft.irfft(np.fft.rfft(np.asarray(prediction[lo:hi],float))*asd,n=N)
    z=hilbert(truth,axis=-1);w=hilbert(p,axis=-1);delta=np.angle(w*z.conj());m=mask[lo:hi]
    peak=np.argmax((abs(z)**2).sum(1),axis=1);inside[lo:hi]=m[np.arange(hi-lo),peak]
    weight=(abs(z)**2)*m[:,None,:]
    phase_gap[lo:hi]=np.sqrt((weight*delta**2).sum((1,2))/weight.sum((1,2)))*180/np.pi
    indices=np.argmax(abs(z),axis=-1)
    phase_peak[lo:hi]=np.abs(np.take_along_axis(delta,indices[:,:,None],axis=-1)[:,:,0])*180/np.pi
    energy[lo:hi]=(truth**2*m[:,None,:]).sum((1,2))/(truth**2).sum((1,2))
    rel[lo:hi]=np.sqrt(((p-truth)**2).sum((1,2))/(truth**2).sum((1,2)))
    f0=np.fft.rfft(truth);df=np.fft.rfft(p-truth)
    ES[lo:hi]=((abs(df[...,1:-1])**2/psd[:,1:-1]).sum((1,2))/(abs(f0[...,1:-1])**2/psd[:,1:-1]).sum((1,2)))
    for key,x in [('target',truth),('noise',n),('residual',p-truth)]:
        f,v=periodogram(x,fs=fs,window='boxcar',detrend='constant',scaling='density',axis=-1)
        sums[key]+=v.sum(0)
assert inside.sum()==1935
assert abs(phase_peak.mean(0)[0]-3.837647)<1e-5
assert abs(rel.mean()-.13424)<1e-5
pars=np.load(T/'parameters.npy');tau=np.array([get_lm_f0tau(float(p[9]),float(p[10]),2,2)[1] for p in pars]);ratio=(length/fs)/tau
def summary(x):return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),p95=float(np.quantile(x,.95)),min=float(x.min()),max=float(x.max()))
out=dict(gap_phase_peak_inside=dict(all=summary(phase_gap[inside]),snr_ge10=summary(phase_gap[inside&strong])),gap_energy_fraction=dict(all=summary(energy),snr_ge10=summary(energy[strong])),gap_over_tau220=dict(all=summary(ratio),snr_ge10=summary(ratio[strong])),psd_relative_error=dict(all=summary(ES),snr_ge10=summary(ES[strong])),phase_peak_mean=phase_peak.mean(0).tolist(),amplitude_relative_mean=float(rel.mean()))
bins=[0,.25,.5,.75,1.00000001]
out['snr_ge10_gap_fraction_bins']=[]
for a,b in zip(bins[:-1],bins[1:]):
    ix=strong&(energy>=a)&(energy<b)
    if ix.any():out['snr_ge10_gap_fraction_bins'].append(dict(lower=a,upper=min(b,1.),mismatch=summary(np.load(F/'per_example_test.npz')['variable__optimized'][ix])))
np.savez_compressed(P/'additional_diagnostics.npz',seeds=seeds,snr=snr,inside=inside,phase_gap=phase_gap,gap_energy_fraction=energy,gap_over_tau220=ratio,psd_relative_error=ES)
np.savez_compressed(P/'population_spectra.npz',frequency=f,**{key:value/len(h) for key,value in sums.items()})

# Independent forward/normal-equation/physical-overlap checks for the prior fit.
bp=R/'FillingGap/results_prd_strategy_20261013/training_direction_basis.npz';B=np.load(bp)['vectors'][:,:128]
E=np.zeros((2*N,4));E[:N,0]=E[N:,2]=1/np.sqrt(N);E[:N,1]=E[N:,3]=(-1.)**np.arange(N)/np.sqrt(N)
Q=np.c_[B,E]*np.load(P/'basis_prior_corrected/coefficient_prior_std.npy')[None,:]
phys=np.fft.irfft(np.fft.rfft(Q.T.reshape(132,2,N))*asd,n=N)
stored=np.load(P/'basis_prior_corrected/basis_per_source.npz');lam=json.loads((P/'basis_prior_corrected/basis_selection.json').read_text())['selected_lambda']
checks=[]
for i in [0,123,2231,5999]:
    D=np.fft.irfft(np.fft.rfft(phys*(1-mask[i]))/asd,n=N);df=np.fft.rfft(D);df[...,0]=df[...,-1]=0;D=np.fft.irfft(df,n=N).reshape(132,-1).T
    u=np.fft.irfft(np.fft.rfft(np.asarray(h[i],float)*(1-mask[i])+noise[i])/asd,n=N);uf=np.fft.rfft(u);uf[...,0]=uf[...,-1]=0;y=np.fft.irfft(uf,n=N).reshape(-1)
    gram=D.T@D;rhs=D.T@y
    c=cho_solve(cho_factor(gram+lam*np.eye(132)),rhs)
    relative=float(np.linalg.norm(c-stored['coefficients'][i])/np.linalg.norm(c))
    assert relative<1e-5,relative
    forward1=D@c
    raw=(c@Q.T).reshape(2,N);raw=np.fft.irfft(np.fft.rfft(raw)*asd,n=N)*(1-mask[i]);forward2=np.fft.rfft(raw)/asd;forward2[...,0]=forward2[...,-1]=0;forward2=np.fft.irfft(forward2,n=N).reshape(-1)
    forwarderr=float(np.linalg.norm(forward1-forward2)/np.linalg.norm(forward1));assert forwarderr<1e-9
    physical=np.fft.irfft(np.fft.rfft((c@Q.T).reshape(2,N))*asd,n=N)
    a=np.fft.rfft(np.asarray(h[i],float))[:,1:-1]/np.sqrt(psd[:,1:-1]);b=np.fft.rfft(physical)[:,1:-1]/np.sqrt(psd[:,1:-1]);direct=1-(a.conj()*b).real.sum()/np.sqrt((abs(a)**2).sum()*(abs(b)**2).sum())
    assert abs(direct-stored['linear_mismatch'][i])<1e-6
    checks.append(dict(index=i,relative_coefficient_cholesky_error=relative,forward_error=forwarderr,physical_mismatch_error=float(abs(direct-stored['linear_mismatch'][i]))))
out['linear_fit_independent_checks']=checks
(P/'additional_diagnostics.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
