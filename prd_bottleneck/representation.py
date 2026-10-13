"""Oracle clean-target basis projection as a representation-only diagnostic."""
import numpy as np
from prd_lowrank.settings import ROOT,CACHE,CAL,PROBE,RANK
from prd_physical.common import Preprocess,metric,summary,write,N
OUT=ROOT/'FillingGap/results_prd_bottleneck_20261013'
def main():
    OUT.mkdir(exist_ok=True);psd=np.load(CAL)['psd'];pre=Preprocess(psd,'white');h=np.load(CACHE/'validation_truth.npy').astype(float);white=pre.encode(h).astype(float);f=np.fft.rfft(white);f[...,0]=f[...,-1]=0;white=np.fft.irfft(f,n=N)
    basis=np.load(PROBE/'training_direction_basis.npz')['vectors'][:,:RANK];c=white.reshape(len(h),-1)@basis;oracle=(c@basis.T).reshape(-1,2,N);f0=np.fft.rfft(pre.encode(h));ends=np.zeros_like(f0);ends[...,0]=f0[...,0];ends[...,-1]=f0[...,-1];ep=np.fft.irfft(ends,n=N)
    mm=metric(h,pre.decode(oracle+ep),psd);old=np.load(ROOT/'FillingGap/results_prd_lowrank_20261013/noise_augmented/validation_prediction.npy');oldmm=metric(h,pre.decode(old),psd);rho=np.load(OUT/'validation_snr.npz')['full_signal'];groups={}
    for lo,hi in [(0,10),(10,20),(20,50),(50,np.inf)]:
        ix=(rho>=lo)&(rho<hi);groups[str(lo)+'_'+str(hi)]=dict(n=int(ix.sum()),oracle=summary(mm[ix]),current=summary(oldmm[ix]))
    write(OUT/'representation_floor.json',dict(passed=True,n=len(h),oracle_scope='projection coefficients from true clean validation waveform; representation diagnostic requiring target; deployment requires inferred coefficients',oracle_projection=summary(mm),current_fixed66=summary(oldmm),groups=groups,basis_rank=RANK,basis_training_only=True,current_endpoint_restored_from_target_for_oracle=True,test_read=False))
    print('oracle mismatch',mm.mean(),'frozen model',oldmm.mean())
if __name__=='__main__':main()
