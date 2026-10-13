"""Frozen fixed66 validation probes; targets appear only in diagnostic metrics."""
import time,json,numpy as np,torch,h5py
from prd_physical.common import N,Preprocess,metric,summary,write
from prd_lowrank.settings import ROOT,CACHE,CAL,PROBE,RANK
from prd_lowrank.common import load_backbone,inferred,check_sources,status,ram_guard
from prd_lowrank.data import packet,reconstruct
from prd_strategy.latent import CoefficientAdapter
from prd_physical.prepare import PARENT,crop_meta
OUT=ROOT/'FillingGap/results_prd_bottleneck_20261013'
RUN=ROOT/'FillingGap/results_prd_lowrank_20261013/noise_augmented'

def zero_ends(x):
    f=np.fft.rfft(x,axis=-1);f[...,0]=f[...,-1]=0
    return np.fft.irfft(f,n=N,axis=-1)

def weighted_encode(pre,x):
    f=np.fft.rfft(x,axis=-1)/pre.asd;f[...,0]=f[...,-1]=0
    return np.fft.irfft(f,n=N,axis=-1)

def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(2);check_sources();ram_guard(OUT)
    cal=np.load(CAL);pre=Preprocess(cal['psd'],'white')
    h=np.load(CACHE/'validation_truth.npy');original=np.load(CACHE/'validation_mask.npy');pars=np.load(CACHE/'validation_parameters.npy');seeds=np.load(CACHE/'validation_seeds.npy');n=np.empty_like(h)
    with h5py.File(PARENT/'validation.h5','r') as parent:
        for i in range(len(h)):
            s,m=crop_meta(pars[i],int(seeds[i]));assert np.array_equal(m,original[i]);n[i]=parent['noise'][i,:,s:s+N]
    rho=np.load(OUT/'validation_snr.npz')['full_signal'];basis=np.load(PROBE/'training_direction_basis.npz')['vectors'][:,:RANK];b32=torch.tensor(basis,dtype=torch.float32)
    model=load_backbone();head=CoefficientAdapter(RANK).eval();head.load_state_dict(torch.load(RUN/'best.pth',map_location='cpu',weights_only=True)['state_dict'])
    predictions={1.:np.load(RUN/'validation_prediction.npy')};noise_report={};t=time.monotonic()
    for factor in (0.,.5):
        u=pre.encode(h*(1-original[:,None])+n*np.float32(factor));p=[]
        for lo in range(0,len(h),16):
            hi=min(lo+16,len(h));y=inferred(model,u[lo:hi],original[lo:hi]);v={k:torch.from_numpy(a) for k,a in packet(u[lo:hi],y,original[lo:hi],basis,np.ones(hi-lo)).items()}
            with torch.inference_mode():p.append(reconstruct(head,v,b32).numpy())
            if lo%256==0:status(OUT,'noise_sensitivity',noise_factor=factor,completed=hi,total=len(h))
        predictions[factor]=np.concatenate(p)
    for factor,p in predictions.items():
        mm=metric(h,pre.decode(p),cal['psd']);groups={}
        for lo,hi in [(0,10),(10,20),(20,50),(50,np.inf)]:
            ix=(rho>=lo)&(rho<hi);groups[str(lo)+'_'+str(hi)]=summary(mm[ix])
        noise_report[str(factor)]=dict(all=summary(mm),groups=groups)
    write(OUT/'noise_sensitivity.json',dict(passed=True,validation_only=True,n=len(h),frozen_model=True,noise_factors=[0.,.5,1.],weighting_psd_fixed=True,gap_samples=66,target_and_gap_unchanged=True,noise_factor0_scope='counterfactual noise-free input to a model trained on noisy observations; joint gap and distribution response diagnostic',results=noise_report,test_read=False))
    # Penalized consistency of a DGF prior with observed G*h+n. G acts in
    # physical units before whitening. All input endpoint bins are excluded.
    prior=predictions[1.].astype(float);u=np.load(CACHE/'validation_input.npy').astype(float);physical_basis=pre.decode(basis.T.reshape(RANK,2,N))
    eig=np.load(PROBE/'training_direction_basis.npz')['eigenvalues'][:RANK];assert (eig>0).all();root=np.sqrt(eig)
    kinds=[('isotropic',x) for x in (1.,10.,100.)]+[('direction_covariance',x) for x in (.01,.1,1.,10.,100.,1000.)]
    corrected={str(kind)+'_'+str(alpha):np.empty_like(prior) for kind,alpha in kinds}
    for i in range(len(h)):
        mask=original[i];A=weighted_encode(pre,physical_basis*(1-mask[None,None])).reshape(RANK,-1).T
        cleanprior=zero_ends(prior[i]);cp=cleanprior.reshape(-1)@basis;ep=prior[i]-cleanprior
        endpoint_response=weighted_encode(pre,pre.decode(ep)*(1-mask[None]))
        obs=zero_ends(u[i]).reshape(-1)-endpoint_response.reshape(-1);residual=obs-A@cp
        G=A.T@A;rhs=A.T@residual;e,V=np.linalg.eigh(G);q=V.T@rhs
        T=A*root[None,:];Gt=T.T@T;rt=T.T@residual;et,Vt=np.linalg.eigh(Gt);qt=Vt.T@rt
        # Observation-conditioned DGF coefficient norm estimates scale;
        # clean parameters and true SNR are excluded from the correction.
        amp2=max(float(cp@cp),1.)
        for kind,alpha in kinds:
            if kind=='isotropic':delta=V@(q/(np.maximum(e,0)+alpha))
            else:delta=root*(Vt@(qt/(np.maximum(et,0)+alpha/amp2)))
            corrected[kind+'_'+str(alpha)][i]=((cp+delta)@basis.T).reshape(2,N)+ep
        if (i+1)%100==0:status(OUT,'observation_consistency',completed=i+1,total=len(h))
    reports={};per={}
    for name,p in [('unchanged_prior',prior),*corrected.items()]:
        mm=metric(h,pre.decode(p),cal['psd']);per[name]=mm;reports[name]=dict(all=summary(mm),groups={})
        for lo,hi in [(0,10),(10,20),(20,50),(50,np.inf)]:
            ix=(rho>=lo)&(rho<hi);reports[name]['groups'][str(lo)+'_'+str(hi)]=summary(mm[ix])
    best=min(corrected,key=lambda k:reports[k]['all']['mean']);delta=per[best]-per['unchanged_prior'];rng=np.random.default_rng(2026101387);boot=np.array([rng.choice(delta,len(h),replace=True).mean() for _ in range(3000)])
    np.savez_compressed(OUT/'consistency_per_validation_source.npz',seeds=seeds,**per)
    result=dict(passed=True,validation_only=True,n=len(h),gap_samples=66,candidates=kinds,selected_from_validation=best,results=reports,paired_delta=dict(mean=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist()),basis_training_only=True,target_used_in_correction=False,true_snr_used_in_correction=False,mask_operator='physical G before white operator W; observation endpoints excluded',likelihood_scope='diagonal calibrated finite-crop PSD approximation; residual correction with a DGF-centered quadratic prior',training_pipeline_changed=False,new_independent_test_required=True,selection_bootstrap_scope='descriptive conditional bootstrap after 9-candidate validation selection; independent confirmation pending',elapsed_seconds=time.monotonic()-t,test_read=False)
    write(OUT/'consistency_probe.json',result);status(OUT,'probe_completed',selected=best,prior_mean=reports['unchanged_prior']['all']['mean'],corrected_mean=reports[best]['all']['mean'],noise_report=noise_report)
if __name__=='__main__':main()
