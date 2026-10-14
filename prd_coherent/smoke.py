import gc,json,time,numpy as np,torch
from prd_physical.common import Preprocess,write
from .settings import OUT,CAL,VARIANTS,SEED
from .data import Data
from .model import CoherentDGF,GapOperator,optimizer
from .objective import Objective
from .common import configure

def main():
    configure();torch.manual_seed(SEED);data=Data('validation','phase_balanced');v=data.batch(np.arange(8));op=GapOperator().cuda();pre=Preprocess(np.load(CAL)['psd'],'white')
    x=torch.randn_like(v['input']);z=torch.randn_like(x);m=v['mask']
    a=(op(x,m)*z).sum();b=(x*op.adjoint(z,m)).sum();adj=float(abs(a-b)/max(float(abs(a)),float(abs(b)),1.))
    ref=pre.encode(pre.decode(x.cpu().numpy())*(1-m.cpu().numpy()[:,None]));actual=op.physical_mask(x,m).detach().cpu().numpy();err=float(abs(ref-actual).max())
    assert adj<3e-5 and err<2e-4,(adj,err)
    report=dict(adjoint_relative_error=adj,physical_gap_operator_max_error=err,variants={});reference=None;lossfn=Objective()
    for variant in VARIANTS:
        model=CoherentDGF(variant).cuda().eval();opt=optimizer(model)
        with torch.no_grad():p=model(v['input'],v['q'],m,v['bands'],v['q_low'])
        if reference is None:reference=p.clone()
        identity=float((p-reference).abs().max());assert identity<1e-5
        model.training_modes();loss=lossfn(model(v['input'],v['q'],m,v['bands'],v['q_low']),v['target'],m);loss.backward()
        grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert len(grads)>0 and all(bool(torch.isfinite(g).all()) for g in grads)
        if variant=='data_consistency':assert model.dc.strength.grad is not None
        if variant=='encoder_adaptation':
            assert any(p.grad is not None and bool((p.grad!=0).any()) for p in model.net.backbone.dgf.backbone.dgf.base.encoder.QtranTimeMixer.parameters())
            assert not any(p.requires_grad for l in model.net.backbone.dgf.backbone.dgf.base.encoder.transformer.layers for p in l.self_attn.parameters())
        report['variants'][variant]=dict(initial_identity_max_error=identity,loss=float(loss),trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),gradient_tensors=len(grads))
        del model,opt,p,grads,loss;gc.collect();torch.cuda.empty_cache()
    # Measure the most expensive encoder adaptation, including Adam states.
    model=CoherentDGF('encoder_adaptation').cuda();opt=optimizer(model);model.training_modes();timings=[]
    torch.cuda.reset_peak_memory_stats();v=data.batch(np.arange(256)%data.count)
    for _ in range(3):
        torch.cuda.synchronize();t=time.monotonic();opt.zero_grad(set_to_none=True)
        loss=lossfn(model(v['input'],v['q'],v['mask'],v['bands'],v['q_low']),v['target'],v['mask']);loss.backward();opt.step();torch.cuda.synchronize();timings.append(time.monotonic()-t)
    report.update(batch=256,batch_seconds=timings,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,passed=True)
    assert report['peak_allocated_gib']<7.0,report
    write(OUT/'smoke.json',report);print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
