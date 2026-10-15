import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import sys,json,hashlib
from pathlib import Path
import numpy as np,torch
from scipy.linalg import cho_factor,cho_solve
P=Path(__file__).resolve().parent;R=Path('/home/ljq/code/RingdownGapFilling_PRD');sys.path.insert(0,str(R))
from prd_physical.common import Preprocess,metric,sha
from prd_lowrank.common import legacy_features
from prd_frequency.features import features
from prd_variable.data import variable_mask
from prd_conditioned.model import ConditionedDGF
from prd_tail.model import TailModel
from prd_tail.prepare import inputs
from prd_tail.settings import START,CAL,OUT
from prd_tail.objective import Objective
from prd_tail.common import analytic

torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.5)
F=OUT;T=F/'fresh_test';ids=np.arange(128);seeds=np.load(T/'seeds.npy');h=np.load(T/'truth.npy',mmap_mode='r')[ids];noise=np.load(T/'noise.npy',mmap_mode='r')[ids];mask=variable_mask(np.load(T/'mask.npy')[ids],seeds[ids],410000000);psd=np.load(CAL)['psd'];pre=Preprocess(psd,'white')
base=ConditionedDGF('noise_phase').cuda().eval();base.load_state_dict(torch.load(START,map_location='cuda',weights_only=True)['state_dict'])
ck=F/'full/tail_unet/best.pth';model=TailModel('tail_unet').cuda().eval();model.load_state_dict(torch.load(ck,map_location='cuda',weights_only=True)['state_dict'])
u=pre.encode(h*(1-mask[:,None])+noise);x=torch.from_numpy(u).cuda()
with torch.inference_mode():
 v=dict(input=x,q=legacy_features(u).cuda(),q_low=torch.from_numpy(features(u)).cuda(),mask=torch.from_numpy(mask).cuda(),bands=base.input_bands(x));packet=inputs(base,v);packet['mask']=packet['mask'].float();p=model(packet)[0]
stored=np.load(F/'variable_optimized_prediction.npy',mmap_mode='r')[ids];err=abs(p.cpu().numpy()-stored);relative=np.linalg.norm(p.cpu().numpy()-stored)/np.linalg.norm(stored)
assert relative<2e-5,relative
nbase=sum(q.numel() for q in base.parameters());nres=sum(q.numel() for q in model.parameters());assert nres==646658 and nbase==9589685
# Reconstruct the manuscript objective independently from the operator definition.
with torch.inference_mode():
 y=torch.from_numpy(pre.encode(h)).cuda();packet['target']=y;actual=float(Objective()(p,packet,{},5));transfer=model.transfer
 P0=torch.fft.rfft(p);H=torch.fft.rfft(y);a=P0[...,1:-1];b=H[...,1:-1];tt=abs(b).square().sum((1,2)).clamp_min(1e-12);pp=abs(a).square().sum((1,2)).clamp_min(1e-12)
 mm=1-(a.conj()*b).real.sum((1,2))/(tt*pp).sqrt();ES=abs(a-b).square().sum((1,2))/tt
 yp=torch.fft.irfft(P0*transfer,n=1056);yt=torch.fft.irfft(H*transfer,n=1056);e=(yp-yt).square();Et=((1+2*packet['mask'][:,None])*e).sum((1,2))/yt.square().sum((1,2)).clamp_min(1e-12)
 z=analytic(yt);w=analytic(yp);A=abs(z);Amax=A.amax(-1,keepdim=True).clamp_min(1e-6);peak=A.square().sum(1).argmax(-1);reg=(A>=.01*Amax)&(torch.arange(1056,device='cuda')[None,None]>=peak[:,None,None]);ratio=(yp-yt)/(A+.03*Amax);Huber=torch.where(abs(ratio)<.5,ratio.square(),abs(ratio)-.25);Env=(Huber*reg).sum((1,2))/reg.sum((1,2)).clamp_min(1)
 wt=reg*(A/Amax).sqrt();den=((abs(w).square()+1e-6*A.square()).sqrt()*A.clamp_min(1e-6)).clamp_min(1e-12);Ph=(wt*(1-(w*z.conj()).real/den)).sum((1,2))/wt.sum((1,2)).clamp_min(1e-6);rho2=2/1056*tt
 expected=float((mm+.2*ES+.02*Et+.5*rho2/(rho2+100)*(.04*Env+.03*Ph)).mean());assert abs(actual-expected)<1e-7
# Independent Cholesky checks of four new-test linear-basis fits.
h=h.astype(float);noise=noise.astype(float)
N=1056;B=np.load(R/'FillingGap/results_prd_strategy_20261013/training_direction_basis.npz')['vectors'][:,:128];E=np.zeros((2112,4));E[:N,0]=E[N:,2]=1/np.sqrt(N);E[:N,1]=E[N:,3]=(-1.)**np.arange(N)/np.sqrt(N);Q=np.c_[B,E]*np.load(P/'basis_prior_corrected/coefficient_prior_std.npy')[None]
asd=pre.asd;phys=np.fft.irfft(np.fft.rfft(Q.T.reshape(132,2,N))*asd,n=N);storedfit=np.load(P/'basis_prior_corrected/basis_per_source.npz');checks=[]
for j in [0,17,63,127]:
 i=ids[j]
 design=np.fft.irfft(np.fft.rfft(phys*(1-mask[j]))/asd,n=N);f=np.fft.rfft(design);f[...,0]=f[...,-1]=0;D=np.fft.irfft(f,n=N).reshape(132,-1).T
 y=np.fft.irfft(np.fft.rfft(h[j]*(1-mask[j])+noise[j])/asd,n=N);f=np.fft.rfft(y);f[...,0]=f[...,-1]=0;y=np.fft.irfft(f,n=N).ravel();c=cho_solve(cho_factor(D.T@D+.001*np.eye(132)),D.T@y)
 ce=float(np.linalg.norm(c-storedfit['coefficients'][i])/np.linalg.norm(c));assert ce<1e-6
 est=np.fft.irfft(np.fft.rfft((c@Q.T).reshape(2,N))*asd,n=N);me=float(abs(metric(h[j],est,psd)-storedfit['linear_mismatch'][i]));assert me<1e-10;checks.append(dict(index=int(i),coefficient_relative_error=ce,physical_mismatch_error=me))
report=dict(base_sha256=sha(START),residual_sha256=sha(ck),prediction_reproduction_relative_error=float(relative),prediction_max_absolute_error=float(err.max()),model_parameter_counts=dict(intermediate=nbase,residual=nres,total=nbase+nres),loss_formula=dict(code=actual,independent=expected,absolute_error=abs(actual-expected)),linear_independent_checks=checks)
(P/'verification_numerical.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
