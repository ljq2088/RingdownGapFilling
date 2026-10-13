"""Validation-only temporal coverage and gap-duration transfer; frozen production code unchanged."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'2')
os.environ.setdefault('NUMBA_NUM_THREADS','1')
import time,json
from pathlib import Path
import numpy as np,torch,h5py
from prd_physical.common import N,FS,Preprocess,metric,summary,write,sha
from prd_physical.prepare import PARENT,crop_meta
from prd_lowrank.settings import CACHE,CAL,PROBE,RANK,SNAPSHOT_SHA
from prd_lowrank.common import load_backbone,inferred
from prd_lowrank.data import packet,reconstruct
from prd_strategy.latent import CoefficientAdapter
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'FillingGap/results_prd_lowrank_20261013'
OUT=ROOT/'FillingGap/results_prd_window_audit_20261013'
LENGTHS=(66,100,132,210)

def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(2)
    h=np.load(CACHE/'validation_truth.npy',mmap_mode='r').astype(float)
    original_mask=np.load(CACHE/'validation_mask.npy',mmap_mode='r')
    params=np.load(CACHE/'validation_parameters.npy');seeds=np.load(CACHE/'validation_seeds.npy')
    n=np.empty_like(h);coverage=[];local_peak=[];starts=[]
    with h5py.File(PARENT/'validation.h5','r') as f:
        assert np.array_equal(seeds,f['seeds'][:len(h)])
        for i in range(len(h)):
            start,m=crop_meta(params[i],int(seeds[i]));full=f['truth'][i].astype(float)
            assert np.array_equal(full[:,start:start+N],h[i]) and np.array_equal(m,original_mask[i])
            assert np.array_equal(params[i],f['parameters'][i])
            n[i]=f['noise'][i,:,start:start+N].astype(float)
            coverage.append(float((h[i]*h[i]).sum()/(full*full).sum()))
            local_peak.append(float(params[i,13]-start));starts.append(start)
    coverage=np.array(coverage);post=(N-1-np.array(local_peak))/FS;tau=params[:,12]
    cal=np.load(CAL);pre=Preprocess(cal['psd'],'white')
    H=np.fft.rfft(h);snr=np.sqrt((4/(FS*N)*(abs(H[...,1:-1])**2/cal['psd'][:,1:-1])).sum((1,2)))
    temporal=dict(n=len(h),parent_samples=32768,parent_seconds=32768/FS,input_samples=N,input_seconds=N/FS,dt_seconds=1/FS,gap_lengths_tested=list(LENGTHS),longest_tau_seconds=summary(tau),tau_quantiles=dict(p10=float(np.quantile(tau,.1)),p50=float(np.quantile(tau,.5)),p90=float(np.quantile(tau,.9))),tau_gt_input_window_n=int((tau>N/FS).sum()),tau_gt_input_window_percent=100*float((tau>N/FS).mean()),post_peak_duration_seconds=summary(post),post_peak_at_least_one_longest_tau_n=int((post>=tau).sum()),post_peak_at_least_three_longest_tau_n=int((post>=3*tau).sum()),post_peak_at_least_five_longest_tau_n=int((post>=5*tau).sum()),joint_AE_unweighted_time_squared_norm_retained=summary(coverage),coverage_quantiles=dict(p10=float(np.quantile(coverage,.1)),p50=float(np.quantile(coverage,.5)),p90=float(np.quantile(coverage,.9))),coverage_scope='unweighted joint A/E fractional-frequency time squared norm fraction in the 528-s crop relative to its 16384-s parent; separate from PSD-weighted SNR or radiated energy',strata={})
    for label,ix in [('mass_lt5e5',params[:,0]<5e5),('mass_5e5_2e6',(params[:,0]>=5e5)&(params[:,0]<2e6)),('mass_ge2e6',params[:,0]>=2e6),('snr_ge10',snr>=10)]:
        temporal['strata'][label]=dict(n=int(ix.sum()),retained_norm=summary(coverage[ix]),tau=summary(tau[ix]))
    write(OUT/'temporal_coverage.json',dict(passed=True,scope='existing1500 validation sources, parent and crop bitwise reproduced',**temporal))
    backbone=load_backbone();basis64=np.load(PROBE/'training_direction_basis.npz')['vectors'][:,:RANK];basis=torch.tensor(basis64,dtype=torch.float32)
    selected=json.loads((RUN/'selection_frozen.json').read_text());chosen=next(x for x in selected['candidates'] if x['variant']=='noise_augmented');assert sha(RUN/'noise_augmented/best.pth')==chosen['checkpoint_sha256']
    ck=torch.load(RUN/'noise_augmented/best.pth',map_location='cpu',weights_only=True);head=CoefficientAdapter(RANK).eval();head.load_state_dict(ck['state_dict'])
    rows=[];per={};t=time.monotonic()
    for length in LENGTHS:
        # All lengths are even: preserve the same physical gap center exactly.
        mask=np.zeros_like(original_mask);a=original_mask.argmax(-1)-(length-66)//2
        assert (a>=0).all() and (a+length<=N).all()
        for i,start in enumerate(a):mask[i,start:start+length]=1
        # Match the frozen validation builder's float32 physical combination.
        u=pre.encode(h.astype('f4')*(1-mask[:,None])+n.astype('f4'))
        if length==66:assert np.array_equal(u,np.load(CACHE/'validation_input.npy',mmap_mode='r'))
        F=np.fft.rfft(h*(1-mask[:,None]));gated_snr=np.sqrt((4/(FS*N)*(abs(F[...,1:-1])**2/cal['psd'][:,1:-1])).sum((1,2)))
        p=[];base=[]
        for lo in range(0,len(h),16):
            hi=min(lo+16,len(h));y=inferred(backbone,u[lo:hi],mask[lo:hi]);base.append(y)
            values={k:torch.from_numpy(v) for k,v in packet(u[lo:hi],y,mask[lo:hi],basis64,np.ones(hi-lo)).items()}
            with torch.inference_mode():p.append(reconstruct(head,values,basis).numpy())
        y=np.concatenate(p);old=np.concatenate(base);physical=pre.decode(y);previous=pre.decode(old)
        mm=metric(h,physical,cal['psd']);oldmm=metric(h,previous,cal['psd']);tm=metric(h,physical);l2=np.sqrt(((h-physical)**2).sum((1,2))/(h*h).sum((1,2)))
        row=dict(gap_samples=length,gap_seconds=length/FS,gap_fraction=length/N,all_rows=summary(mm),frequency_snapshot=summary(oldmm),time=summary(tm),relative_l2=summary(l2),snr_ge10=summary(mm[snr>=10]),snr_ge20=summary(mm[snr>=20]),gated_snr_ge10=summary(mm[gated_snr>=10]),gated_snr_ge20=summary(mm[gated_snr>=20]),gap_center_fixed=True,elapsed_seconds=time.monotonic()-t)
        rows.append(row);per[str(length)+'__psd']=mm
        write(OUT/'progress.json',dict(rows=rows));print(json.dumps(row),flush=True)
    # The original-duration pass reproduces the production validation metric.
    error=abs(rows[0]['all_rows']['mean']-chosen['validation']['score']);assert error<1e-7,error
    write(OUT/'gap_transfer_results.json',dict(passed=True,validation_only=True,n=len(h),model_train_gap_samples=66,backbone_sha=SNAPSHOT_SHA,head_sha=chosen['checkpoint_sha256'],original_duration_mean_reproduction_error=error,physics_and_time_grid_unchanged=True,gap_center_held_fixed=True,rows=rows,test_predictions_read=False,scope='frozen model extrapolation to longer signal-suppression intervals on existing validation population; independent test and variable-duration training required for a new primary claim'))
    np.savez_compressed(OUT/'per_source.npz',seeds=seeds,snr=snr,longest_tau=tau,retained_time_squared_norm=coverage,**per)
if __name__=='__main__':main()
