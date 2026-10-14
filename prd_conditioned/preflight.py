import copy,json,time,gc,numpy as np,torch
from prd_physical.common import Preprocess,metric,sha,write,N
from prd_physical.physics import Simulator
from prd_lowrank.common import atomic_checkpoint
from .settings import *
from .model import ConditionedDGF,packet,reconstruct,phase_features,context_features,ResidualHead
from .objective import Objective

def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(2)
    if not (OUT/'noise_calibration.npz').exists():
        sim=Simulator();pre=Preprocess(np.load(CAL)['psd'],'white');sums=[]
        for seed in range(2027100000,2027100128):
            u=pre.encode(sim.noise(seed)[:,6000:6000+N]);f=np.fft.rfft(u);f[:,0]=f[:,-1]=0;v=np.fft.irfft(f,n=N)
            sums.append(np.mean(v*v,axis=-1))
        rms=np.sqrt(np.mean(sums,axis=0));assert np.isfinite(rms).all() and (rms>0).all()
        np.savez(OUT/'noise_calibration.npz',rms=rms,noise_seeds=[2027100000,2027100127],n=128,window_start=6000,excluded_endpoints=True)
    torch.cuda.set_per_process_memory_fraction(.78);torch.manual_seed(SEED)
    model=ConditionedDGF('noise_phase').cuda().eval()
    v={k:torch.tensor(np.load(PREVIOUS_OUT/'cache'/'validation0'/(k+'.npy'),mmap_mode='r')[:16],device='cuda') for k in FIELDS}
    target=torch.tensor(np.load(CACHE/'validation_target.npy')[:16],device='cuda');truth=np.load(CACHE/'validation_truth.npy')[:16];pre=Preprocess(np.load(CAL)['psd'],'white')
    with torch.inference_mode():
        base=model.backbone(v['input'],v['q'],v['mask'],v['bands'],v['q_low']);values=packet(v['input'],base,v['mask'],v['q_low'],model.backbone.adapter.basis,model.noise_rms)
        y=reconstruct(model.head,values,model.backbone.adapter.basis)
        initial_error=float(abs(metric(truth,pre.decode(y.cpu().numpy()),pre.psd)-metric(truth,pre.decode(base.cpu().numpy()),pre.psd)).max())
        assert initial_error<2e-6,initial_error
        direct=1-((torch.fft.rfft(y).conj()*torch.fft.rfft(target)).real[...,1:-1].sum((1,2)))/(torch.fft.rfft(y).abs().square()[...,1:-1].sum((1,2))*torch.fft.rfft(target).abs().square()[...,1:-1].sum((1,2))).sqrt()
        metric_error=float(abs(direct.cpu().numpy()-metric(truth,pre.decode(y.cpu().numpy()),pre.psd)).max());assert metric_error<2e-6,metric_error
        low=v['q_low'].clone();low[:,:,1]+=2*torch.pi
        phase_error=float((phase_features(low)-phase_features(v['q_low'])).abs().max());assert phase_error<3e-6
        changed=context_features(2*v['input'],2*base,model.noise_rms);sensitivity=float((changed-values['context']).abs().max());assert sensitivity>.1
    # Values consumed by autograd must be regular tensors, rather than inference tensors.
    values={k:a.clone() for k,a in values.items()};basis=model.backbone.adapter.basis.detach().clone()
    head=ResidualHead('noise_phase').cuda().train();opt=torch.optim.AdamW(head.parameters(),lr=3e-4);lossfn=Objective()
    def step(head,opt,values,target,mask):
        opt.zero_grad(set_to_none=True);y=reconstruct(head,values,basis);loss=lossfn(y,target,mask);assert torch.isfinite(loss);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.);assert torch.isfinite(norm);opt.step();return float(loss.detach()),float(norm)
    step(head,opt,values,target,v['mask'])
    path=OUT/'preflight_resume.pth';atomic_checkpoint(path,dict(model=head.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()))
    step(head,opt,values,target,v['mask']);expected={k:a.detach().clone() for k,a in head.state_dict().items()}
    fresh=ResidualHead('noise_phase').cuda().train();opt2=torch.optim.AdamW(fresh.parameters(),lr=3e-4)
    c=torch.load(path,map_location='cuda',weights_only=False);fresh.load_state_dict(c['model']);opt2.load_state_dict(c['optimizer']);torch.set_rng_state(c['torch_rng'].cpu());torch.cuda.set_rng_state(c['cuda_rng'].cpu())
    step(fresh,opt2,values,target,v['mask']);assert all(torch.equal(expected[k],a) for k,a in fresh.state_dict().items());path.unlink()
    # Head benchmark uses repeated examples strictly for timing.
    batch={k:a.repeat((64,)+(1,)*(a.ndim-1)) for k,a in values.items()};targ=target.repeat(64,1,1);mask=v['mask'].repeat(64,1)
    torch.cuda.synchronize();t=time.monotonic()
    for _ in range(5):step(head,opt,batch,targ,mask)
    torch.cuda.synchronize();head_seconds=(time.monotonic()-t)/5
    # A representative selective-finetuning step checks the exact modules used by the long run.
    from .finetune import optimizer
    for k in values:del batch[k]
    del head,fresh,opt,opt2;gc.collect();torch.cuda.empty_cache()
    live_opt=optimizer(model);model.train();model.backbone.dgf.backbone.dgf.base.encoder.eval();torch.cuda.synchronize();t=time.monotonic()
    for _ in range(3):
        live_opt.zero_grad(set_to_none=True);out=model(v['input'],v['q'],v['mask'],v['bands'],v['q_low']);loss=lossfn(out,target,v['mask']);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(loss) and torch.isfinite(norm);live_opt.step()
    torch.cuda.synchronize();live_seconds=(time.monotonic()-t)/3
    source_manifest={str(a.relative_to(ROOT)):sha(a) for folder in ['prd_conditioned','prd_variable','prd_lowrank','prd_strategy','prd_frequency','prd_optimize','prd_physical','model','utils','config'] for a in (ROOT/folder).rglob('*.py')}
    assets=json.loads((PREVIOUS_OUT/'asset_manifest.json').read_text());assets[str(START.relative_to(ROOT))]=sha(START);assets[str((OUT/'noise_calibration.npz').relative_to(ROOT))]=sha(OUT/'noise_calibration.npz')
    write(OUT/'runtime_source_manifest.json',source_manifest);write(OUT/'asset_manifest.json',assets)
    result=dict(passed=True,zero_initialization_max_direct_mismatch_error=initial_error,white_physical_metric_max_error=metric_error,phase_2pi_periodicity_max_error=phase_error,absolute_noise_context_scale_sensitivity=sensitivity,head_model_optimizer_rng_exact_resume=True,head_batch1024_step_seconds=head_seconds,finetune_batch16_step_seconds=live_seconds,gpu_peak_mib=torch.cuda.max_memory_allocated()/2**20,noise_rms=model.noise_rms.cpu().tolist(),test_read=False)
    write(OUT/'preflight.json',result);print(json.dumps(result))
if __name__=='__main__':main()
