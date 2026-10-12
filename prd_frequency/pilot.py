"""Matched three-epoch feasibility probe; fresh long runs restart frozen warm-start."""
import os
os.environ.setdefault('NUMBA_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','2')
import copy,json,time
import numpy as np,torch
from prd_physical.common import Preprocess,write
from prd_optimize.loss import Objective
from .settings import *
from .model import FrequencyDGF
from .train import validate,optimizer
from .data import load_maps,batches,physical_truth

def main():
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(GPU_FRACTION)
 tr=load_maps('train');va=load_maps('validation');truth=physical_truth('validation');cal=np.load(OLD/'calibration_psd.npz');pre=Preprocess(cal['psd'],'white');objective=Objective(pre,cal,'joint_hann');reports={}
 for kind in VARIANTS:
  torch.manual_seed(SEED);model=FrequencyDGF(kind).cuda();ema=copy.deepcopy(model).eval()
  for p in ema.parameters():p.requires_grad_(False)
  opt=optimizer(model);torch.manual_seed(SEED+1);rng=torch.Generator().manual_seed(SEED+2);rows=[]
  for epoch in range(1,4):
   t=time.monotonic();model.train();order=torch.randperm(12000,generator=rng).numpy();total=0.
   for v in batches(tr,order):
    opt.zero_grad(set_to_none=True);p=model(v['input'],v['q'],v['mask'],v['bands'],v['q_low']);loss=objective(p,v['target'],v['mask']);assert torch.isfinite(loss)
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=float(loss.detach())*len(v['input'])
    with torch.no_grad():
     for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,.005)
     for a,b in zip(ema.buffers(),model.buffers()):a.copy_(b)
   row=dict(epoch=epoch,train_loss=total/12000,online=validate(model,va,truth,pre,cal),ema=validate(ema,va,truth,pre,cal),seconds=time.monotonic()-t)
   rows.append(row);print(json.dumps(dict(variant=kind,**row)),flush=True)
  reports[kind]=rows;write(OUT/'pilot_progress.json',reports)
  del model,ema,opt,v,p,loss;torch.cuda.empty_cache()
 write(OUT/'pilot.json',dict(passed=True,train_n=12000,epochs=3,long_run_restarts_original_checkpoint=True,test_read=False,results=reports))
if __name__=='__main__':main()
