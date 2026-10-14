import json,time,gc,numpy as np,torch
from prd_conditioned.model import ConditionedDGF,packet
from prd_conditioned.data import MixedData
from prd_variable.settings import FIELDS
from prd_physical.common import write,sha
from prd_lowrank.common import status
from .common import analytic,guard,configure,check
from .settings import *

@torch.inference_mode()
def inputs(net,v):
    p=net(v['input'],v['q'],v['mask'],v['bands'],v['q_low'])
    a=packet(v['input'],p,v['mask'],v['q_low'],net.backbone.adapter.basis,net.noise_rms,v.get('sign'))
    f=torch.cat((a['features'],a['context'],a['phase']),-1)
    tr=net._tail_transfer if hasattr(net,'_tail_transfer') else None
    if tr is None:
        from prd_physical.common import Preprocess
        pre=Preprocess(np.load(CAL)['psd'],'white');tr=torch.tensor(pre.asd/pre.sigma.mean(),device=p.device,dtype=p.dtype)
    physical=torch.fft.irfft(torch.fft.rfft(p)*tr,n=1056)
    peak=analytic(physical).abs().square().sum(1).argmax(-1).to(p.dtype)*.5
    return dict(prediction=p,features=f,obs=v['input'],mask=v['mask'].to(torch.uint8),peak=peak,scale=a['scale'],sign=a['sign'])

def main():
    configure();guard();check()
    if (OUT/'prepared.json').exists():return
    net=ConditionedDGF('noise_phase').cuda().eval()
    net.load_state_dict(torch.load(START,map_location='cuda',weights_only=True)['state_dict'])
    np.save(OUT/'basis.npy',net.backbone.adapter.basis.cpu().numpy())
    pars=np.load(CACHE/'train_parameters.npy');stats=dict(log_mass_mean=float(np.log(pars[:,9]).mean()),log_mass_std=float(np.log(pars[:,9]).std()))
    write(OUT/'parameter_normalization.json',stats)
    from pycbc.conversions import get_lm_f0tau
    spin=np.linspace(.4,.8,801);f=[];tau=[]
    for l,m in [(2,2),(2,1),(3,3),(4,4)]:
        pairs=[get_lm_f0tau(1e6,s,l,m) for s in spin];f.append([x[0] for x in pairs]);tau.append([x[1] for x in pairs])
    np.savez(OUT/'qnm_table.npz',spin=spin,frequency=np.array(f).T,tau=np.array(tau).T,reference_mass=1e6)
    for split in ('validation','train'):
        data=MixedData(split);count=data.count
        if split=='train':
            fourth=COHERENT/'mixed_cache'
            assert all(np.load(fourth/(k+'.npy'),mmap_mode='r').shape[0]==TRAIN_N for k in FIELDS)
            data.views.append({k:torch.from_numpy(np.load(fourth/(k+'.npy'),mmap_mode='c')) for k in FIELDS})
            data.ram.append({});data.gpu.append({});data.sign=torch.cat((data.sign,torch.ones((1,count))))
        for j in range(1 if split=='validation' else 4):
            path=OUT/'cache'/f'{split}{j}';path.mkdir(parents=True,exist_ok=True)
            shape=dict(prediction=(2,1056),features=(470,),obs=(2,1056),mask=(1056,),peak=(),scale=(1,),sign=())
            arrays={k:np.load(path/(k+'.npy'),mmap_mode='r+') if (path/(k+'.npy')).exists() else np.lib.format.open_memmap(path/(k+'.npy'),mode='w+',dtype='u1' if k=='mask' else 'f4',shape=(count,)+s) for k,s in shape.items()}
            done=json.loads((path/'progress.json').read_text())['completed'] if (path/'progress.json').exists() else 0
            began=done;t=time.monotonic();last=0
            for lo in range(done,count,256):
                hi=min(lo+256,count);v=data.batch(np.arange(lo,hi)+j*count)
                with torch.inference_mode():values=inputs(net,v)
                for k,a in values.items():
                    a=a.cpu().numpy();assert np.isfinite(a).all();arrays[k][lo:hi]=a
                if time.monotonic()-last>30 or hi==count:
                    for a in arrays.values():a.flush()
                    write(path/'progress.json',dict(completed=hi,total=count))
                    status(OUT,'preparing_baseline_cache',split=split,view=j,completed=hi,total=count,remaining_view_seconds=(count-hi)*(time.monotonic()-t)/max(hi-began,1));last=time.monotonic();guard()
            del arrays
        del data;gc.collect();torch.cuda.empty_cache()
    del net;gc.collect();torch.cuda.empty_cache()
    write(OUT/'prepared.json',dict(training_sources=TRAIN_N,views=4,observations_per_epoch=4*TRAIN_N,validation_sources=VAL_N,test_read=False,baseline_sha256=sha(START),fourth_view='Variance-preserving mixture of original and independent Gaussian noise, with recomputed mask and features; training views may share noise components. Inherited audited generation 0.'))
