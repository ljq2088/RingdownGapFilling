"""CPU-only frozen-backbone pilot: loss balancing and valid noise/sign augmentation."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'2')
import sys,json,time,copy,fcntl
import numpy as np,torch
from prd_physical.common import Preprocess,metric,write,sha,N
from prd_optimize.loss import Objective
from .settings import *
from .latent import CoefficientAdapter,coordinates,RANK
from .loss import StrategyObjective
KINDS=['legacy','balanced','balanced_augmented','balanced_repeated','direct_spectral']

def status(path,stage,**kw):
 v=dict(stage=stage,pid=os.getpid(),updated_unix=time.time(),**kw);write(path/'status.json',v);print(json.dumps(v),flush=True)
def tensors(u,p,mask,target,basis):
 a=coordinates(u,p,mask,basis);a.update(mask=np.asarray(mask,'f4'),target=np.asarray(target,'f4'))
 return {k:torch.from_numpy(v) for k,v in a.items()}
def prediction(model,a,basis):
 model.eval();out=[]
 with torch.no_grad():
  for lo in range(0,len(a['target']),128):
   c=model(a['features'][lo:lo+128],a['coefficients'][lo:lo+128],a['scale'][lo:lo+128]);y=(c@basis.T).reshape(-1,2,N)+a['endpoint'][lo:lo+128];out.append(y.numpy())
 return np.concatenate(out)
def validate(model,a,basis,truth,pre,cal):
 y=prediction(model,a,basis);p=pre.decode(y)
 return dict(score=float(metric(truth,p,cal['psd']).mean()),time=float(metric(truth,p).mean()),relative_l2=float(np.sqrt(((p-truth)**2).sum((1,2))/(truth*truth).sum((1,2))).mean()))
def main(kind,rank=RANK,seed=SEED):
 if kind not in KINDS or rank not in [128,256]:raise ValueError('unknown pilot configuration')
 path=OUT/('head_'+kind+('' if rank==RANK else '_rank'+str(rank))+('' if seed==SEED else '_seed'+str(seed)));path.mkdir(exist_ok=True)
 with open(path/'training.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (path/'selected.json').exists():return
  torch.set_num_threads(2);torch.manual_seed(seed);basis_numpy=np.load(OUT/'training_direction_basis.npz')['vectors'][:,:rank];basis=torch.tensor(basis_numpy,dtype=torch.float32)
  cal=np.load(CAL);pre=Preprocess(cal['psd'],'white');p=np.load(OUT/'frequency_snapshot_train_prediction.npy');u=np.load(CACHE/'train_input.npy',mmap_mode='r')[:PRED_TRAIN_N];mask=np.load(CACHE/'train_mask.npy',mmap_mode='r')[:PRED_TRAIN_N];h=np.load(CACHE/'train_target.npy',mmap_mode='r')[:PRED_TRAIN_N]
  train=tensors(u,p,mask,h,basis_numpy)
  if kind=='balanced_augmented':
   with np.load(OUT/'augmented_training.npz') as f:extra=tensors(f['input'],f['prediction'],f['mask'],f['target'],basis_numpy)
   train={k:torch.cat([train[k],extra[k]]) for k in train}
  if kind=='balanced_repeated':train={k:torch.cat([a,a]) for k,a in train.items()}
  val=tensors(np.load(CACHE/'validation_input.npy',mmap_mode='r'),np.load(OUT/'frequency_snapshot_validation_prediction.npy'),np.load(CACHE/'validation_mask.npy',mmap_mode='r'),np.load(CACHE/'validation_target.npy',mmap_mode='r'),basis_numpy);truth=np.load(CACHE/'validation_truth.npy',mmap_mode='r').astype(float)
  model=CoefficientAdapter(rank);opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4);objective=StrategyObjective(pre,cal,kind!='legacy',auxiliary_weights=(0.,0.) if kind=='direct_spectral' else None)
  # Independent objective reproduction for the inherited recipe.
  original=Objective(pre,cal,'joint_hann',device='cpu');base=(train['coefficients'][:128]@basis.T).reshape(-1,2,N)+train['endpoint'][:128]
  legacy=StrategyObjective(pre,cal,False)(base,train['target'][:128],train['mask'][:128]);expected=original(base,train['target'][:128],train['mask'][:128]);assert abs(float(legacy-expected))<1e-6
  initial=validate(model,val,basis,truth,pre,cal);best=initial['score'];stale=0;best_epoch=0;history=[];rng=torch.Generator().manual_seed(seed+1);best_state=copy.deepcopy(model.state_dict());status(path,'initial',kind=kind,n_train=len(train['target']),n_unique_sources=PRED_TRAIN_N,initial=initial,parameters=sum(p.numel() for p in model.parameters()))
  for epoch in range(1,61):
   t=time.monotonic();model.train();total=0.;order=torch.randperm(len(train['target']),generator=rng)
   for lo in range(0,len(order),128):
    idx=order[lo:lo+128];v={k:a[idx] for k,a in train.items()};opt.zero_grad(set_to_none=True);coeff=model(v['features'],v['coefficients'],v['scale']);y=(coeff@basis.T).reshape(-1,2,N)+v['endpoint'];loss=objective(y,v['target'],v['mask']);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=float(loss.detach())*len(idx)
   result=validate(model,val,basis,truth,pre,cal)
   if result['score']<best-1e-6:best=result['score'];best_epoch=epoch;stale=0;best_state=copy.deepcopy(model.state_dict())
   else:stale+=1
   row=dict(epoch=epoch,train_loss=total/len(order),validation=result,best=best,best_epoch=best_epoch,stale=stale,epoch_seconds=time.monotonic()-t);history.append(row);write(path/'history.json',history);status(path,'epoch_complete',**row)
   if epoch>=10 and stale>=8:break
  model.load_state_dict(best_state);result=validate(model,val,basis,truth,pre,cal);torch.save(dict(state_dict=best_state,basis=basis,rank=rank,kind=kind,training_seed=seed,best_epoch=best_epoch,score=best,training_sources=PRED_TRAIN_N,augmented_observations=(kind=='balanced_augmented'),test_read=False),path/'best.pth');np.save(path/'validation_prediction.npy',prediction(model,val,basis));write(path/'selected.json',dict(passed=True,kind=kind,finished_epoch=epoch,best_epoch=best_epoch,metrics=result,checkpoint_sha256=sha(path/'best.pth'),test_read=False,scope='frozen epoch25 frequency backbone; 4096-source CPU coefficient-adapter feasibility pilot'))
  status(path,'completed',best_epoch=best_epoch,result=result,test_read=False)
if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]) if len(sys.argv)>2 else RANK,int(sys.argv[3]) if len(sys.argv)>3 else SEED)
