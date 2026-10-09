"""Pre-training gates on training/validation only; short steps are not performance claims."""
import os
os.environ.setdefault("OMP_NUM_THREADS","2")
import copy,json,time
import h5py,numpy as np,torch
from prd_physical.common import Preprocess,metric,write,sha,N
from prd_physical.run import BASE,load
from prd_physical.model import PhysicalDGF
from .settings import *
from .model import OptimizedDGF
from .loss import Objective
from .data import load_maps,batches,physical_truth
from .train import optimizer,validate

def deterministic_training_loss(model,tr,objective):
 model.eval();values=[]
 with torch.no_grad():
  for batch in batches(tr,np.arange(len(tr["input"]))):
   out=model(batch["input"],batch["q"],batch["mask"],batch["bands"])
   values.append(float(objective(out,batch["target"],batch["mask"])))
 return float(np.mean(values))

def main():
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(GPU_FRACTION)
 cal=np.load(OLD/"calibration_psd.npz");pre=Preprocess(cal["psd"],"white")
 va=load_maps("validation");truth=physical_truth("validation");results={}
 # No writes are allowed into mapped data; all source tensors are gathered into batches.
 assert len(va["input"])==VAL_N
 with h5py.File(OLD/"validation_white_features.h5","r") as f:
  for k in ["input","target","q"]:assert np.array_equal(va[k].numpy(),f[k][:])
 torch.manual_seed(SEED);model=OptimizedDGF("joint_hann").cuda().eval()
 v=next(batches(va,np.arange(8)))
 with torch.no_grad():
  before=model.dgf(v["input"],v["q"],v["mask"],v["bands"]);after=model(v["input"],v["q"],v["mask"],v["bands"])
 assert torch.equal(before,after),"joint zero head must preserve previous reconstruction"
 obj=Objective(pre,cal,"joint_hann")
 physical=pre.decode(after.cpu().numpy());h=truth[:8];hh=np.hanning(N)
 a=torch.fft.irfft(torch.fft.rfft(after)*obj.transfer,n=N)*obj.hann
 b=torch.fft.irfft(torch.fft.rfft(v["target"])*obj.transfer,n=N)*obj.hann
 A,B=torch.fft.rfft(a),torch.fft.rfft(b);cross=((A*B.conj()).real*obj.hw).sum((-2,-1));aa=(A.abs()**2*obj.hw).sum((-2,-1));bb=(B.abs()**2*obj.hw).sum((-2,-1))
 actual=(1-cross/(aa*bb).sqrt()).cpu().numpy();expected=metric(h*hh,physical*hh,cal["hann_psd"])
 assert np.max(abs(actual-expected))<2e-5
 del model;torch.cuda.empty_cache()
 # Equal original train prefix, same unmodified inputs; full validation at start/end.
 tr={}
 for k in ["input","target","q"]:
  with h5py.File(OLD/"train_white_features.h5","r") as f:tr[k]=torch.from_numpy(f[k][:512])
 for k,file in [("mask","train_crops.h5"),("bands","train_white_db4_bands.h5")]:
  with h5py.File(OLD/file,"r") as f:tr[k]=torch.from_numpy(f[k][:512])
 for kind in VARIANTS:
  torch.manual_seed(SEED);model=OptimizedDGF(kind).cuda();objective=Objective(pre,cal,kind);opt=optimizer(model)
  initial=validate(model,va,truth,pre,cal);initial_fixed_loss=deterministic_training_loss(model,tr,objective);torch.manual_seed(SEED+1);losses=[];t=time.monotonic();head_grad=None
  model.train()
  for step in range(24):
   ix=np.arange((step%4)*128,(step%4+1)*128);batch=next(batches(tr,ix));opt.zero_grad(set_to_none=True)
   out=model(batch["input"],batch["q"],batch["mask"],batch["bands"]);loss=objective(out,batch["target"],batch["mask"]);assert torch.isfinite(loss)
   loss.backward()
   if model.joint is not None and step==0:
    head_grad=float(model.joint.head.weight.grad.norm());assert head_grad>0
   torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
  final=validate(model,va,truth,pre,cal)
  final_fixed_loss=deterministic_training_loss(model,tr,objective)
  print(json.dumps(dict(variant=kind,initial_fixed_loss=initial_fixed_loss,final_fixed_loss=final_fixed_loss)),flush=True)
  assert final_fixed_loss<initial_fixed_loss,(kind,"fixed eval-mode training loss did not decrease",initial_fixed_loss,final_fixed_loss)
  results[kind]=dict(initial_validation=initial,after_24_training_steps_validation=final,fixed_training_loss_before=initial_fixed_loss,fixed_training_loss_after=final_fixed_loss,first_four_loss_mean=float(np.mean(losses[:4])),last_four_loss_mean=float(np.mean(losses[-4:])),seconds=time.monotonic()-t,head_gradient_first_step=head_grad,parameters=sum(x.numel() for x in model.parameters()),gpu_peak_mb=torch.cuda.max_memory_allocated()/2**20)
  print(json.dumps({kind:results[kind]}),flush=True);del model,opt;torch.cuda.empty_cache()
 write(OUT/"preflight.json",dict(passed=True,validation_feature_prefix_bitwise_identical=True,joint_initial_prediction_bitwise_identical=True,hann_torch_metric_max_error=float(np.max(abs(actual-expected))),variants=results,new_test_read=False,scope="short learning probe; not a held-out improvement claim"))
if __name__=="__main__":main()
