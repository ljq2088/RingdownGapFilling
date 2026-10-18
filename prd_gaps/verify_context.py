"""Operator equivalence, augmentation isolation, and GPU precision/gradient checks."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from .online import SourceBank,event_masks
from .masks import propagate_tdi
from .context_model import ContextReconstructor,focused_loss
from .metrics import mismatch

def run(root):
    torch.set_num_threads(4);torch.manual_seed(743)
    data=SourceBank(root,'train',8);c=data.config;rng=np.random.default_rng(456)
    for i in range(20):
        events=[dict(link='all' if i%4==0 else int(rng.choice(c['links'])),
                     start_s=float(rng.uniform(0,c['n']/c['fs']-250)),duration_s=float(rng.uniform(1,200)))]
        links=np.ones((6,c['parent_n']),bool)
        for e in events:
            a=int(np.floor(e['start_s']*c['fs']))+c['pad'];b=int(np.ceil((e['start_s']+e['duration_s'])*c['fs']))+c['pad']
            for j,link in enumerate(c['links']):
                if e['link'] in ('all',link):links[j,a:b]=False
        _,reference=propagate_tdi(links,c['links'],c['tdi_combinations'],c['fs'],c['arm_seconds'],c['order'])
        assert np.array_equal(event_masks(events,c),reference[:,c['pad']:c['pad']+c['n']]),i
    a=data[0];b=data[0];assert all(torch.equal(x,y) for x,y in zip(a[:4],b[:4]))
    data.set_epoch(1);b=data[0];assert torch.equal(a[1],b[1]) and not torch.equal(a[0],b[0])
    assert (a[0][:2][a[2]==0]==0).all()
    x,y,v,info=[torch.stack([data[i][j] for i in range(2)]) for j in range(4)]
    x,y,v=x.cuda(),y.cuda(),v.cuda();psd=torch.from_numpy(data.psd.astype('float32')).cuda()
    result=dict(mask_equivalence_cases=20,deterministic_views=True,epoch_refresh=True,models={})
    for kind in ['attention','spectral']:
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();m=ContextReconstructor(kind).cuda();m.train()
        start=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.bfloat16):p=m(x)
        loss=focused_loss(p,y,v,info,psd);loss.backward();torch.cuda.synchronize()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        assert all(p.grad is not None for p in m.parameters() if p.requires_grad)
        seconds=time.perf_counter()-start;m.eval()
        with torch.no_grad():
            fp=m(x).float().cpu().numpy()
            with torch.autocast('cuda',dtype=torch.bfloat16):bf=m(x).float().cpu().numpy()
        report=dict(parameters=sum(p.numel() for p in m.parameters()),loss=float(loss),batch2_forward_backward_seconds=seconds,
                    peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,
                    fp32_bf16_relative_rms=float(np.linalg.norm(fp-bf)/np.linalg.norm(fp)),
                    fp32_bf16_mismatch=float(np.mean(mismatch(bf,fp,data.psd))),all_parameters_receive_gradients=True)
        result['models'][kind]=report;print(json.dumps({kind:report}),flush=True)
        del m,p,loss
    destination=Path(__file__).parent/'reports/context_verification.json';destination.write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);run(p.parse_args().data)
