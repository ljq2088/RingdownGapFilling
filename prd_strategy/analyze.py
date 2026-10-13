"""Train-derived postprocessing ablations and output-gradient audit; test remains sealed."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'2')
import json,time
import numpy as np,torch
from prd_physical.common import Preprocess,metric,summary,write,sha,N,FS
from .settings import *

def status(stage,**kw):
 v=dict(stage=stage,pid=os.getpid(),updated_unix=time.time(),**kw);write(OUT/'analysis_status.json',v);print(json.dumps(v),flush=True)
def remove_endpoints(y):
 f=np.fft.rfft(np.asarray(y,float),axis=-1);f[...,0]=f[...,-1]=0;return np.fft.irfft(f,n=N,axis=-1)
def score_work(h,p):
 cross=(h*p).sum((1,2));den=np.sqrt((h*h).sum((1,2))*(p*p).sum((1,2)))
 return 1-np.clip(np.divide(cross,den,out=np.zeros(len(h)),where=den>0),-1,1)
def fit_basis():
 file=OUT/'training_direction_basis.npz'
 if file.exists():
  f=np.load(file);return f['vectors'],f['eigenvalues']
 a=np.load(CACHE/'train_target.npy',mmap_mode='r');cov=np.zeros((2*N,2*N));count=len(a);t=time.monotonic();last=0
 for lo in range(0,count,512):
  z=remove_endpoints(a[lo:lo+512]).reshape(-1,2*N);norm=np.linalg.norm(z,axis=1);assert (norm>0).all();z/=norm[:,None];cov+=z.T@z
  if time.monotonic()-last>30 or lo+512>=count:status('training_direction_covariance',completed=min(count,lo+512),total=count,elapsed_seconds=time.monotonic()-t);last=time.monotonic()
 eigenvalues,vectors=np.linalg.eigh(cov/count);idx=np.argsort(eigenvalues)[::-1];eigenvalues=eigenvalues[idx];vectors=vectors[:,idx[:512]]
 np.savez_compressed(file,vectors=vectors,eigenvalues=eigenvalues,train_count=count,centered=False,uniform_per_source_direction_weight=True);return vectors,eigenvalues

def fit_spectral(h,p,u,ridge,include_observation):
 H=np.fft.rfft(h,axis=-1);P=np.fft.rfft(p,axis=-1);U=np.fft.rfft(u,axis=-1)
 norm=np.sqrt((abs(H[...,1:-1])**2).sum((1,2)));H/=norm[:,None,None]
 X=np.concatenate([P,U],axis=1) if include_observation else P;X=X/norm[:,None,None]
 cxx=np.einsum('iaf,ibf->fab',X,X.conj())/len(h);chx=np.einsum('iaf,ibf->fab',H,X.conj())/len(h)
 d=X.shape[1];tr=cxx.trace(axis1=1,axis2=2).real/d
 reg=ridge*tr+max(tr.max(),1e-20)*1e-10
 gain=chx@np.linalg.inv(cxx+reg[:,None,None]*np.eye(d));gain[0]=0;gain[-1]=0
 gain[0,:,:2]=gain[-1,:,:2]=np.eye(2)
 return gain

def apply_spectral(gain,p,u,include_observation):
 X=np.concatenate([np.fft.rfft(p,axis=-1),np.fft.rfft(u,axis=-1)],axis=1) if include_observation else np.fft.rfft(p,axis=-1)
 return np.fft.irfft(np.einsum('fab,ibf->iaf',gain,X),n=N,axis=-1)

def full_summary(truth,y,pre,cal,snr):
 p=pre.decode(y);mm=metric(truth,p,cal['psd']);hann=np.hanning(N)
 rel=np.sqrt(((p-truth)**2).sum((1,2))/(truth*truth).sum((1,2)))
 return dict(psd=summary(mm),time=summary(metric(truth,p)),hann=summary(metric(truth*hann,p*hann,cal['hann_psd'])),relative_l2=summary(rel),snr_lt20=summary(mm[snr<20]),snr_ge20=summary(mm[snr>=20]))

def gradients(p,h,mask,pre,cal):
 torch.set_num_threads(2);p=torch.tensor(p,dtype=torch.float64,requires_grad=True);h=torch.tensor(h,dtype=torch.float64);m=torch.tensor(mask,dtype=torch.float64)
 w=torch.tensor(pre.weights(),dtype=torch.float64);transfer=torch.tensor(pre.asd/pre.sigma.mean(),dtype=torch.float64)
 P,H=torch.fft.rfft(p),torch.fft.rfft(h);tt=(H.abs().square()*w).sum((1,2));pp=(P.abs().square()*w).sum((1,2));cross=((H*P.conj()).real*w).sum((1,2))
 mm=(1-cross/(tt*pp).clamp_min(1e-24).sqrt()).mean();spec=.5*(((P-H).abs().square()*w).sum((1,2))/tt.clamp_min(1e-12)).mean()
 a=torch.fft.irfft(P*transfer,n=N);b=torch.fft.irfft(H*transfer,n=N);err=(a-b).square();den=b.square().sum((1,2)).clamp_min(1e-12)
 timegap=.5*((err.sum((1,2))+2*(err*m[:,None]).sum((1,2)))/den).mean()
 hann=torch.tensor(np.hanning(N));hw=pre.sigma.mean()**2/cal['hann_psd'];hw=hw.copy();hw[:,0]=hw[:,-1]=0;hw=torch.tensor(hw/hw.mean())
 A,B=torch.fft.rfft(a*hann),torch.fft.rfft(b*hann);bt=(B.abs().square()*hw).sum((1,2));ap=(A.abs().square()*hw).sum((1,2));c=((B*A.conj()).real*hw).sum((1,2));rel=((A-B).abs().square()*hw).sum((1,2))/bt.clamp_min(1e-12)
 auxiliary=.15*(1-c/(bt*ap).clamp_min(1e-24).sqrt()+.5*(torch.sqrt(1+rel)-1)).mean()
 components={'direct_mismatch':mm,'spectral_mse':spec,'time_gap_mse':timegap,'hann_auxiliary':auxiliary};g={k:torch.autograd.grad(v,p,retain_graph=True)[0].detach().numpy().reshape(len(p),-1) for k,v in components.items()};base=g['direct_mismatch'];bn=np.linalg.norm(base,axis=1);rows={}
 for k,grad in g.items():
  norm=np.linalg.norm(grad,axis=1);cos=(grad*base).sum(1)/np.maximum(norm*bn,1e-300)
  rows[k]=dict(value=float(components[k].detach()),output_gradient_norm=float(np.linalg.norm(grad)),median_per_source_norm_ratio_to_primary=float(np.median(norm/np.maximum(bn,1e-300))),median_cosine_with_primary=float(np.median(cos)),fraction_opposite_to_primary=float((cos<0).mean()))
 return rows

def main():
 OUT.mkdir(exist_ok=True);cal=np.load(CAL);pre=Preprocess(cal['psd'],'white');truth=np.load(CACHE/'validation_truth.npy',mmap_mode='r').astype(float);h=np.load(CACHE/'validation_target.npy',mmap_mode='r').astype(float);hb=remove_endpoints(h)
 hf=np.fft.rfft(truth);snr=np.sqrt((4/(FS*N)*(abs(hf[...,1:-1])**2/cal['psd'][:,1:-1])).sum((1,2)));u=np.load(CACHE/'validation_input.npy',mmap_mode='r')
 train_h=np.load(CACHE/'train_target.npy',mmap_mode='r')[:PRED_TRAIN_N].astype(float);train_u=np.load(CACHE/'train_input.npy',mmap_mode='r')[:PRED_TRAIN_N].astype(float)
 vectors,eigenvalues=fit_basis();rank_grid=[16,32,64,96,128,192,256,384,512];mix_grid=[.25,.5,1.];freq=np.fft.rfftfreq(N,1/FS);reports={};gradient={};selected={};candidates_all={};predicted={}
 for anchor in ['previous_joint','frequency_snapshot']:
  p=np.load(OUT/(anchor+'_validation_prediction.npy')).astype(float);pb=remove_endpoints(p);baseline=full_summary(truth,p,pre,cal,snr);identity=score_work(hb,pb);assert abs(identity.mean()-baseline['psd']['mean'])<1e-7
  train_p=np.load(OUT/(anchor+'_train_prediction.npy')).astype(float);candidates=[];best_by_family={};best_arrays={}
  def candidate(family,name,y,**kwargs):
   mm=score_work(hb,remove_endpoints(y));row=dict(family=family,name=name,mean=float(mm.mean()),**kwargs);candidates.append(row)
   if family not in best_by_family or row['mean']<best_by_family[family]['mean']:best_by_family[family]=row;best_arrays[family]=y
  candidate('baseline','identity',p)
  # Fixed smoothing of predictions; target waveform parameters never enter inference.
  for fc in [.025,.04,.06,.08,.12,.2,.3]:
   for order in [4,8]:
    gain=1/(1+(freq/fc)**order);gain[0]=gain[-1]=1
    filtered=np.fft.irfft(np.fft.rfft(p)*gain,n=N)
    for mix in [.5,1.]:candidate('smooth_frequency',f'fc{fc}_order{order}_mix{mix}',p+mix*(filtered-p),cutoff_hz=fc,order=order,mix=mix)
  flat=pb.reshape(len(p),-1);coeff=flat@vectors;offset=p-pb
  for rank in rank_grid:
   y=(coeff[:,:rank]@vectors[:,:rank].T).reshape(p.shape)+offset
   for mix in mix_grid:candidate('direction_basis',f'rank{rank}_mix{mix}',p+mix*(y-p),rank=rank,mix=mix,training_direction_energy_retained=float(eigenvalues[:rank].sum()/eigenvalues.sum()))
  for include in [False,True]:
   for ridge in [.001,.01,.1]:
    gain=fit_spectral(train_h,train_p,train_u,ridge,include);filtered=apply_spectral(gain,p,u,include)
    for mix in mix_grid:
     family='spectral_p_and_observation' if include else 'spectral_prediction';name=f'{family}_ridge{ridge}_mix{mix}';candidate(family,name,p+mix*(filtered-p),ridge=ridge,mix=mix)
    np.savez_compressed(OUT/(anchor+('_observed' if include else '_predicted')+f'_ridge{ridge}_gain.npz'),gain=gain)
  reports[anchor]={'baseline':baseline,'best_by_family':{}}
  for family,row in best_by_family.items():
   y=best_arrays[family];rep=full_summary(truth,y,pre,cal,snr);rep['configuration']=row;rep['reduction_percent']=100*(baseline['psd']['mean']-rep['psd']['mean'])/baseline['psd']['mean'];reports[anchor]['best_by_family'][family]=rep
   predicted[anchor+'__'+family]=y.astype('f4')
  winner=min(best_by_family,key=lambda k:best_by_family[k]['mean']);selected[anchor]=dict(family=winner,**best_by_family[winner]) if False else best_by_family[winner]
  candidates_all[anchor]=candidates
  # Residual energy localization uses validation targets only for diagnostics.
  R=np.fft.rfft(pb-hb);T=np.fft.rfft(hb);bins=[]
  for lo,hi in [(0,.01),(.01,.03),(.03,.1),(.1,.25),(.25,1.01)]:
   ix=(freq>=lo)&(freq<hi);bins.append(dict(low=lo,high=hi,residual_energy_fraction=float((abs(R[...,ix])**2).sum()/(abs(R)**2).sum()),target_energy_fraction=float((abs(T[...,ix])**2).sum()/(abs(T)**2).sum())))
  reports[anchor]['error_by_frequency']=bins
  ix=np.random.default_rng(SEED).choice(PRED_TRAIN_N,512,replace=False);gradient[anchor]=gradients(train_p[ix],train_h[ix],np.load(CACHE/'train_mask.npy',mmap_mode='r')[ix],pre,cal)
  status('anchor_analyzed',anchor=anchor,best=selected[anchor])
 result=dict(passed=True,n_validation=len(h),all_rows_retained=True,test_read=False,primary_metric='direct unmaximized joint A/E PSD mismatch arithmetic mean',fit_scope=dict(direction_basis_train_n=60000,spectral_fit_train_n=PRED_TRAIN_N,validation_n=VAL_N),anchors=reports,selected_by_validation=selected,selection_requires_new_independent_test=True,current_training_unchanged=True)
 write(OUT/'strategy_results.json',result);write(OUT/'candidate_grid.json',candidates_all);write(OUT/'output_gradient_audit.json',dict(passed=True,scope='gradients with respect to predicted samples on 512 training rows; not parameter-gradient conflicts',anchors=gradient,test_read=False));np.savez_compressed(OUT/'validation_prototype_predictions.npz',**predicted)
 status('completed',selected=selected,test_read=False)
if __name__=='__main__':main()
