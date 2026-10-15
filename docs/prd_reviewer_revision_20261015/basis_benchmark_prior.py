"""CPU-only, validation-selected linear fit in the frozen training basis.

Read-only input assets. No imports of the active training runtime and no CUDA.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import time,json,hashlib
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(2)

P=Path(__file__).resolve().parent/'basis_prior_corrected'; P.mkdir(exist_ok=True)
R=Path('/home/ljq/code/RingdownGapFilling_PRD')
F=R/'FillingGap/results_prd_conditioned_20261014'
C=R/'FillingGap/results_prd_optimization_20261013/cache'
V=R/'FillingGap/results_prd_variable_20261013/cache/validation0'
N=1056;FS=2.;BATCH=24
psd=np.load(R/'FillingGap/results_prd_physical_transfer_20261009/calibration_psd.npz')['psd'].astype(float)
asd=np.sqrt(psd*FS/2);asd[:,0]=asd[:,1];asd[:,-1]=asd[:,-2]
bp=R/'FillingGap/results_prd_strategy_20261013/training_direction_basis.npz'
B=np.load(bp)['vectors'][:,:128].astype(float)
E=np.zeros((2*N,4))
E[:N,0]=1/np.sqrt(N); E[:N,1]=(-1.)**np.arange(N)/np.sqrt(N)
E[N:,2]=1/np.sqrt(N); E[N:,3]=(-1.)**np.arange(N)/np.sqrt(N)
K=132
target=np.load(C/'train_target.npy',mmap_mode='r')
endpoint_variance=np.zeros(4)
for lo in range(0,len(target),512):
    x=np.asarray(target[lo:lo+512],float);f=np.fft.rfft(x);ep=np.stack((f[:,0,0].real,f[:,0,-1].real,f[:,1,0].real,f[:,1,-1].real),axis=1)/np.sqrt(N)
    f[...,0]=f[...,-1]=0;norm=np.sqrt((np.fft.irfft(f,n=N)**2).sum((1,2)))
    endpoint_variance+=((ep/norm[:,None])**2).sum(0)
endpoint_variance/=len(target)
prior_std=np.r_[np.sqrt(np.maximum(np.load(bp)['eigenvalues'][:128],1e-16)),np.sqrt(np.maximum(endpoint_variance,1e-16))]
Q=np.c_[B,E]*prior_std[None,:]
np.save(P/'coefficient_prior_std.npy',prior_std)
phys=np.fft.irfft(np.fft.rfft(Q.T.reshape(K,2,N))*asd,n=N)
grid=np.r_[0.,np.logspace(-6,2,9)]
start=time.monotonic()

def status(stage,**kw):
    d=dict(stage=stage,elapsed=time.monotonic()-start,**kw)
    tmp=P/'basis_status.tmp';tmp.write_text(json.dumps(d,indent=2));tmp.replace(P/'basis_status.json')
    print(json.dumps(d),flush=True)

def white(x):return np.fft.irfft(np.fft.rfft(np.asarray(x,float))/asd,n=N)
def p0(x):
    f=np.fft.rfft(x);f[...,0]=f[...,-1]=0
    return np.fft.irfft(f,n=N)
def mm(h,p):
    cross=(h*p).sum((1,2));den=np.sqrt((h*h).sum((1,2))*(p*p).sum((1,2)))
    return 1-np.clip(np.divide(cross,den,out=np.zeros_like(cross),where=den>0),-1,1)
def summary(a):return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p95=float(np.quantile(a,.95)))
def solve(split,truth,observation,masks,lambdas):
    count=len(truth);scores=np.empty((len(lambdas),count));coeff=np.empty((count,K));oracle={r:np.empty(count) for r in (32,64,128)}
    maxcond=0
    for lo in range(0,count,BATCH):
        hi=min(lo+BATCH,count);m=np.asarray(masks[lo:hi],float)
        # d=G h+n in physical units; W G W^-1 B is the forward design.
        design=white(phys[None,:,:,:]*(1-m[:,None,None,:])).reshape(hi-lo,K,2*N)
        design=p0(design.reshape(hi-lo,K,2,N)).reshape(hi-lo,K,2*N)
        y=p0(observation[lo:hi]).reshape(hi-lo,2*N)
        t=p0(white(truth[lo:hi]));flat=t.reshape(hi-lo,2*N)
        gram=design@design.transpose(0,2,1);rhs=(design@y[...,None])[...,0]
        eig,vec=np.linalg.eigh(gram);eig=np.maximum(eig,0)
        projected=(vec.transpose(0,2,1)@rhs[...,None])[...,0]
        for j,lam in enumerate(lambdas):
            den=eig+lam
            den=np.maximum(den,np.maximum(eig[:,-1:]*1e-10,1e-15))
            c=(vec@(projected/den)[...,None])[...,0]
            prediction=(c@Q.T).reshape(hi-lo,2,N)
            scores[j,lo:hi]=mm(t,p0(prediction))
            if len(lambdas)==1:coeff[lo:hi]=c
        for rank in oracle:
            c=flat@B[:,:rank];projection=c@B[:,:rank].T
            oracle[rank][lo:hi]=mm(t,projection.reshape(hi-lo,2,N))
        if lo%(BATCH*10)==0 or hi==count:
            status('basis_'+split,completed=hi,total=count,seconds_per_source=(time.monotonic()-start)/hi)
    return scores,coeff,oracle

val_h=np.load(C/'validation_truth.npy',mmap_mode='r')
val_u=np.load(V/'input.npy',mmap_mode='r')
val_m=np.load(V/'mask.npy',mmap_mode='r')
scores,_,val_oracle=solve('validation',val_h,val_u,val_m,grid)
selected=int(np.argmin(scores.mean(axis=1)));lam=float(grid[selected])
selection=dict(lambda_grid=grid.tolist(),mean_validation_mismatch=scores.mean(axis=1).tolist(),selected_lambda=lam,selected_using='all 1500 validation sources only',objective='||P0 W(d-G W^-1 Q a)||_2^2 + lambda ||a||_2^2; Q=[B,E] diag(training_direction_coefficient_std)',basis_rank=128,endpoint_nuisance_coefficients=4,white_fourier_weights='same frozen PSD and excluded DC/Nyquist as DGF')
(P/'basis_selection.json').write_text(json.dumps(selection,indent=2))
status('validation_selected',**selection)

test=F/'fresh_test';h=np.load(test/'truth.npy',mmap_mode='r');noise=np.load(test/'noise.npy',mmap_mode='r')
oldm=np.load(test/'mask.npy',mmap_mode='r');seeds=np.load(test/'seeds.npy')
assert not np.isin(seeds,np.load(C/'validation_seeds.npy')).any()
lengths=np.array([64+2*int(np.random.default_rng(int(seed)+410000000).integers(21)) for seed in seeds])
starts=oldm.argmax(-1)-(lengths-66)//2
mask=((np.arange(N)[None,:]>=starts[:,None])&(np.arange(N)[None,:]<starts[:,None]+lengths[:,None])).astype(float)
u=white(h*(1-mask[:,None,:])+noise)
scores,coeff,oracle=solve('test',h,u,mask,np.array([lam]));linear=scores[0]
metrics=np.load(F/'per_example_test.npz');snr=metrics['snr'];dgf=metrics['variable__optimized'];strong=snr>=10
results=dict(protocol=selection,basis_sha256=hashlib.sha256(bp.read_bytes()).hexdigest(),basis_orthogonality_max_error=float(abs(B.T@B-np.eye(128)).max()),test_n=len(h),linear=dict(all=summary(linear),snr_ge10=summary(linear[strong])),dgf=dict(all=summary(dgf),snr_ge10=summary(dgf[strong])),oracle_basis={str(rank):dict(all=summary(a),snr_ge10=summary(a[strong])) for rank,a in oracle.items()})
rng=np.random.default_rng(2026101551)
for key,ix in [('all',np.ones(len(h),bool)),('snr_ge10',strong)]:
    delta=dgf[ix]-linear[ix]
    boot=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(5000)])
    results.setdefault('paired_dgf_minus_linear',{})[key]=dict(mean=float(delta.mean()),percentile_ci95=np.quantile(boot,[.025,.975]).tolist(),resamples=5000)
np.savez_compressed(P/'basis_per_source.npz',seeds=seeds,snr=snr,linear_mismatch=linear,dgf_mismatch=dgf,coefficients=coeff,**{'oracle_rank'+str(r):a for r,a in oracle.items()})
(P/'basis_results.json').write_text(json.dumps(results,indent=2))
status('basis_benchmark_completed',results=results)
