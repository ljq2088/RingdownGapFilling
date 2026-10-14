import gc,json,numpy as np,torch
from prd_physical.common import write
from prd_lowrank.common import status
from .settings import *
from .data import Data
from .model import TailModel
from .objective import Objective
from .validation import evaluate,validate
from .common import configure,check

def main():
    configure();check();pred=np.load(OUT/'cache/validation0/prediction.npy')
    baseline=evaluate(pred);assert abs(baseline['score']-.007923060569647502)<1e-8
    write(OUT/'baseline_validation.json',baseline);reports={}
    for variant in VARIANTS:
        model=TailModel(variant).cuda();data=Data('validation',variant);v=data.batch(np.arange(8));model.train();p,aux=model(v)
        identity=float((p-v['prediction']).abs().max());assert identity==0,(variant,identity)
        loss=Objective()(p,v,aux,10);loss.backward()
        assert bool(torch.isfinite(loss)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        active=sum(p.numel() for p in model.parameters() if p.grad is not None and p.grad.abs().max()>0)
        assert active>0
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4);optimizer.step();model.clip()
        with torch.no_grad():p,_=model(v);assert bool(torch.isfinite(p).all())
        reports[variant]=dict(initial_identity_max_error=identity,finite_backward=True,nonzero_gradient_parameters=active,parameters=sum(p.numel() for p in model.parameters()),smoke_loss=float(loss.detach()))
        del model,data,v,p,loss,optimizer;gc.collect();torch.cuda.empty_cache()
    # Exercise the actual training batch for the largest activation footprint.
    model=TailModel('tail_unet').cuda();data=Data('validation','tail_unet');data.cache();v=data.batch(np.arange(128));torch.cuda.reset_peak_memory_stats()
    p,aux=model(v);loss=Objective()(p,v,aux,10);loss.backward();opt=torch.optim.AdamW(model.parameters());opt.step()
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    reports['batch128_peak_gib']=torch.cuda.max_memory_allocated()/2**30
    write(OUT/'preflight.json',dict(passed=True,checks=reports,baseline_validation=baseline,test_read=False));status(OUT,'preflight_passed',batch128_peak_gib=reports['batch128_peak_gib'])
    del model,data,v,p,loss,opt;gc.collect();torch.cuda.empty_cache()
