"""Shared-parameter loss gradients at the frozen fixed-gap checkpoint."""
import numpy as np,torch,json
from prd_variable.model import VariableDGF
from prd_strategy.loss import StrategyObjective
from prd_physical.common import N,Preprocess,write
from prd_lowrank.settings import ROOT,CACHE,CAL
OUT=ROOT/'FillingGap/results_prd_bottleneck_20261013'

def components(p,h,mask,obj):
    P,H=torch.fft.rfft(p),torch.fft.rfft(h);tt=(H.abs().square()*obj.w).sum((1,2));pp=(P.abs().square()*obj.w).sum((1,2));c=((H*P.conj()).real*obj.w).sum((1,2))
    mm=1-c/(tt*pp).clamp_min(1e-24).sqrt();spec=((P-H).abs().square()*obj.w).sum((1,2))/tt.clamp_min(1e-12)
    a=torch.fft.irfft(P*obj.transfer,n=N);b=torch.fft.irfft(H*obj.transfer,n=N);e=(a-b).square();den=b.square().sum((1,2)).clamp_min(1e-12);time=(e.sum((1,2))+2*(e*mask[:,None]).sum((1,2)))/den
    A,B=torch.fft.rfft(a*obj.hann),torch.fft.rfft(b*obj.hann);t=(B.abs().square()*obj.hw).sum((1,2));q=(A.abs().square()*obj.hw).sum((1,2));cross=((B*A.conj()).real*obj.hw).sum((1,2));rel=((A-B).abs().square()*obj.hw).sum((1,2))/t.clamp_min(1e-12);hm=1-cross/(t*q).clamp_min(1e-24).sqrt();amp=2*(torch.sqrt(1+rel)-1)
    return dict(direct=mm.mean(),spectral=.5*spec.mean(),time_gap=.05*time.mean(),hann=.02*(hm+.25*amp).mean())

def main():
    torch.set_num_threads(2);torch.manual_seed(2026101381);model=VariableDGF().cpu().eval();cal=np.load(CAL);obj=StrategyObjective(Preprocess(cal['psd'],'white'),cal,True)
    rho=np.load(OUT/'train_snr.npz')['full_signal'];maps={k:np.load(CACHE/('train_'+k+'.npy'),mmap_mode='r') for k in ('input','q','q_low','bands','mask','target')};params=tuple(model.parameters());groups={};cursor=0
    for name,p in model.named_parameters():
        group='coefficients' if name.startswith('adapter.') else 'frequency' if name.startswith('dgf.frequency.') else 'joint' if name.startswith('dgf.backbone.joint.') else 'mask_refiner' if '.refiner.' in name else 'original_dgf'
        groups.setdefault(group,[]).append((cursor,cursor+p.numel()));cursor+=p.numel()
    result={}
    for lo,hi in [(0,10),(10,20),(20,50),(50,np.inf)]:
        ix=np.flatnonzero((rho>=lo)&(rho<hi))[:16];v={k:torch.from_numpy(np.array(a[ix],copy=True)) for k,a in maps.items()};p=model(v['input'],v['q'],v['mask'],v['bands'],v['q_low']);terms=components(p,v['target'],v['mask'],obj)
        error=abs(float((sum(terms.values())-obj(p,v['target'],v['mask'])).detach()));assert error<1e-6
        gradient={}
        for term,value in terms.items():
            g=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True);flat=torch.cat([torch.zeros_like(p).reshape(-1) if a is None else a.reshape(-1) for a,p in zip(g,params)]).double();gradient[term]=flat
            assert torch.isfinite(flat).all()
        def norm(g):return float(g.norm())
        def cosine(a,b):return float(a.dot(b)/(a.norm()*b.norm()).clamp_min(1e-300))
        row=dict(n=len(ix),indices=ix.tolist(),loss_sum_equivalence_error=error,terms={})
        for term,g in gradient.items():
            row['terms'][term]=dict(weighted_value=float(terms[term].detach()),gradient_norm=norm(g),ratio_to_direct=norm(g)/norm(gradient['direct']),cosine_to_direct=cosine(g,gradient['direct']),groups={})
            for name,spans in groups.items():
                a=torch.cat([g[start:end] for start,end in spans]);b=torch.cat([gradient['direct'][start:end] for start,end in spans]);row['terms'][term]['groups'][name]=dict(norm=norm(a),direct_norm=norm(b),ratio_to_direct=norm(a)/max(norm(b),1e-300),cosine_to_direct=cosine(a,b))
        result[str(lo)+'_'+str(hi)]=row
    write(OUT/'parameter_gradient_audit.json',dict(passed=True,scope='16 training sources per SNR stratum, fixed66, frozen old DGF+epoch374 coefficient adapter; deterministic eval forward; derivatives of the actual .5/.05/.02 training losses; weights unchanged',joint_training_pipeline_changed=False,test_read=False,results=result))
    for k,row in result.items():print(k,{t:(v['ratio_to_direct'],v['cosine_to_direct']) for t,v in row['terms'].items()})
if __name__=='__main__':main()
