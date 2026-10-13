"""Compute-only concurrent CPU timing; weights discarded, no model selection."""
import time,copy,sys
import numpy as np,torch
from prd_physical.common import Preprocess,write
from prd_strategy.latent import CoefficientAdapter
from prd_strategy.loss import StrategyObjective
from .settings import *
from .data import TrainingData,reconstruct

def main(label):
    torch.set_num_threads(2);torch.manual_seed(SEED)
    data=TrainingData(OUT/'preflight_run/cache',128,'noise_augmented')
    basis=torch.tensor(np.load(PROBE/'training_direction_basis.npz')['vectors'][:,:RANK],dtype=torch.float32)
    model=CoefficientAdapter(RANK);ema=copy.deepcopy(model);opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    cal=np.load(CAL);objective=StrategyObjective(Preprocess(cal['psd'],'white'),cal,True)
    times=[]
    for i in range(160):
        t=time.monotonic();v=data.batch((np.arange(BATCH)+i*BATCH)%(128*VIEWS));opt.zero_grad(set_to_none=True)
        loss=objective(reconstruct(model,v,basis),v['target'],v['mask']);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        with torch.no_grad():
            for old,new in zip(ema.parameters(),model.parameters()):old.lerp_(new,1-EMA)
        if i>=16:times.append(time.monotonic()-t)
    value=dict(passed=True,concurrent_benchmark=True,weights_discarded=True,threads=2,mean_batch_seconds=float(np.mean(times)),estimated_epoch_compute_seconds=float(np.mean(times)*np.ceil(TRAIN_N*VIEWS/BATCH)))
    write(OUT/('benchmark_'+label+'.json'),value);print(value)
if __name__=='__main__':main(sys.argv[1])
