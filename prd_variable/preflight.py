import numpy as np,torch,copy,time,json
from prd_physical.common import Preprocess,metric,sha,write
from prd_lowrank.common import inferred,load_backbone,check_sources,atomic_checkpoint,status
from prd_lowrank.data import packet,reconstruct
from prd_strategy.latent import CoefficientAdapter
from prd_strategy.loss import StrategyObjective
from prd_frequency.features import features
from .settings import *
from .model import VariableDGF
from .data import variable_mask,Dataset

def objective():
    cal=np.load(CAL);obj=StrategyObjective(Preprocess(cal['psd'],'white'),cal,True)
    for k in ('w','transfer','hann','hw'):setattr(obj,k,getattr(obj,k).cuda())
    return obj

def main():
    OUT.mkdir(exist_ok=True);check_sources();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.78)
    model=VariableDGF().cuda();cal=np.load(CAL);pre=Preprocess(cal['psd'],'white')
    u=np.load(CACHE/'validation_input.npy')[:16];mask=np.load(CACHE/'validation_mask.npy')[:16];target=np.load(CACHE/'validation_target.npy')[:16];truth=np.load(CACHE/'validation_truth.npy')[:16]
    v={k:torch.tensor(np.load(CACHE/('validation_'+k+'.npy'))[:16],device='cuda') for k in ('input','q','q_low','mask','bands')}
    model.eval()
    with torch.no_grad():p=model(v['input'],v['q'],v['mask'],v['bands'],v['q_low']).cpu().numpy()
    base=load_backbone();old=inferred(base,u,mask);head=CoefficientAdapter(RANK);head.load_state_dict(torch.load(HEAD,map_location='cpu',weights_only=True)['state_dict'])
    basis=np.load(PROBE/'training_direction_basis.npz')['vectors'][:,:RANK]
    packet0={k:torch.from_numpy(a) for k,a in packet(u,old,mask,basis,np.ones(16)).items()}
    with torch.no_grad():ref=reconstruct(head,packet0,torch.tensor(basis,dtype=torch.float32)).numpy()
    error=float(abs(metric(truth,pre.decode(p),cal['psd'])-metric(truth,pre.decode(ref),cal['psd'])).max());assert error<2e-6,error
    P=torch.fft.rfft(torch.tensor(p));H=torch.fft.rfft(torch.tensor(target));num=(P.conj()*H).real[...,1:-1].sum((1,2));den=(P.abs().square()[...,1:-1].sum((1,2))*H.abs().square()[...,1:-1].sum((1,2))).sqrt();direct=(1-num/den).numpy()
    equivalence=float(abs(direct-metric(truth,pre.decode(p),cal['psd'])).max());assert equivalence<2e-6
    t=time.monotonic();lossfn=objective();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-6)
    model.train();optimizer.zero_grad(set_to_none=True);y=model(v['input'],v['q'],v['mask'],v['bands'],v['q_low']);loss=lossfn(y,torch.tensor(target,device='cuda'),v['mask']);loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(loss) and torch.isfinite(norm);optimizer.step();torch.cuda.synchronize()
    generator=torch.Generator().manual_seed(SEED);state=generator.get_state().clone();saved=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=state)
    atomic_checkpoint(OUT/'preflight_resume.pth',saved);restored=torch.load(OUT/'preflight_resume.pth',map_location='cpu',weights_only=False)
    assert torch.equal(restored['rng'],state)
    for k,value in saved['model'].items():assert torch.equal(value.cpu(),restored['model'][k])
    for index,entry in saved['optimizer']['state'].items():
        for k,value in entry.items():
            if torch.is_tensor(value):assert torch.equal(value.cpu(),restored['optimizer']['state'][index][k])
    random=variable_mask(np.load(CACHE/'validation_mask.npy'),np.load(CACHE/'validation_seeds.npy'),410000000);assert set(random.sum(-1).astype(int))==set(LENGTHS)
    write(OUT/'preflight.json',dict(passed=True,frozen_start_max_mismatch_error=error,white_physical_metric_max_error=equivalence,finite_loss=float(loss.detach()),finite_gradient=float(norm),checkpoint_model_optimizer_rng_roundtrip=True,gap_bounds_verified=True,gpu_batch16_step_seconds=time.monotonic()-t,gpu_peak_mib=torch.cuda.max_memory_allocated()/2**20,test_read=False))
    (OUT/'preflight_resume.pth').unlink();status(OUT,'preflight_passed',**json.loads((OUT/'preflight.json').read_text()))
if __name__=='__main__':main()
