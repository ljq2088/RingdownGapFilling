import json,time,numpy as np,torch
from prd_physical.common import N,FS,Preprocess,metric,write,sha
from prd_physical.physics import Simulator
from prd_physical.prepare import crop_meta
from prd_lowrank.common import legacy_features,status
from prd_frequency.features import features
from prd_variable.data import variable_mask,fixed_mask
from .settings import *
from .model import CoherentDGF
from .validation import metrics,phase
from .common import check,configure,resources

def main():
    check();configure();resources();batch=json.loads((OUT/'protocol.json').read_text())['batch']
    frozen=json.loads((OUT/'selection_frozen.json').read_text());selected=frozen['selection'];ckpath=selected['checkpoint'];assert sha(ckpath)==selected['checkpoint_sha256']
    test=OUT/'fresh_test';test.mkdir(exist_ok=True);seeds=np.arange(TEST_SEED,TEST_SEED+TEST_N)
    for split in ('train','validation'):assert not np.isin(seeds,np.load(CACHE/(split+'_seeds.npy'))).any()
    for old in (2030200000,2032200000,2034200000,2036200000,2038200000):assert not ((seeds>=old)&(seeds<old+6000)).any()
    shapes=dict(truth=(TEST_N,2,N),noise=(TEST_N,2,N),mask=(TEST_N,N),parameters=(TEST_N,15));maps={}
    for k,shape in shapes.items():
        p=test/(k+'.npy');maps[k]=np.load(p,mmap_mode='r+') if p.exists() else np.lib.format.open_memmap(p,mode='w+',dtype='f8' if k=='parameters' else 'f4',shape=shape)
    progress=test/'progress.json';done=json.loads(progress.read_text())['completed'] if progress.exists() else 0
    sim=Simulator();t=time.monotonic();begin=done
    for i in range(done,TEST_N):
        h,n,pars=sim.example(int(seeds[i]));start,m=crop_meta(pars,int(seeds[i]));maps['truth'][i]=h[:,start:start+N];maps['noise'][i]=n[:,start:start+N];maps['parameters'][i]=pars;maps['mask'][i]=m
        if (i+1)%100==0 or i+1==TEST_N:
            for a in maps.values():a.flush()
            write(progress,dict(completed=i+1,total=TEST_N));status(OUT,'generating_untouched_test',completed=i+1,total=TEST_N,remaining_seconds=(TEST_N-i-1)*(time.monotonic()-t)/max(i+1-begin,1));resources()
    np.save(test/'seeds.npy',seeds)
    models={'previous':CoherentDGF('phase_balanced').cuda().eval(),'optimized':CoherentDGF(selected['variant']).cuda().eval()}
    models['optimized'].load_state_dict(torch.load(ckpath,map_location='cuda',weights_only=True)['state_dict'])
    psd=np.load(CAL)['psd'];pre=Preprocess(psd,'white');h=maps['truth'].astype(float);f=np.fft.rfft(h)[...,1:-1];rho=np.sqrt((4/(FS*N)*abs(f)**2/psd[:,1:-1]).sum((1,2)))
    reports={};per={};rng=np.random.default_rng(SEED+100)
    for family,m in [('variable',variable_mask(maps['mask'],seeds,410000000)),('fixed66',fixed_mask(maps['mask'],66)),('fixed100',fixed_mask(maps['mask'],100))]:
        predictions={name:np.lib.format.open_memmap(OUT/(family+'_'+name+'_prediction.npy'),mode='w+',dtype='f4',shape=(TEST_N,2,N)) for name in models}
        for lo in range(0,TEST_N,batch):
            hi=min(lo+batch,TEST_N);u=pre.encode(maps['truth'][lo:hi]*(1-m[lo:hi,None])+maps['noise'][lo:hi]);q=legacy_features(u).cuda();low=torch.from_numpy(features(u)).cuda();x=torch.from_numpy(u).cuda();mask=torch.from_numpy(m[lo:hi]).cuda()
            with torch.inference_mode():
                bands=models['previous'].input_bands(x)
                for name,model in models.items():predictions[name][lo:hi]=model(x,q,mask,bands,low).cpu().numpy()
            if lo%1024==0:status(OUT,'evaluating_untouched_test',family=family,completed=hi,total=TEST_N)
        group={}
        for name,pred in predictions.items():
            pred.flush();p=pre.decode(pred);mm=metric(h,p,psd);pe=phase(h,p);per[family+'__'+name]=mm;per[family+'__'+name+'__phase']=pe;group[name]=metrics(h,p,psd)
            A=np.fft.rfft(h)[...,1:-1]/np.sqrt(psd[:,1:-1]);B=np.fft.rfft(p)[...,1:-1]/np.sqrt(psd[:,1:-1]);audit=1-np.clip((A.conj()*B).real.sum((1,2))/np.sqrt((abs(A)**2).sum((1,2))*(abs(B)**2).sum((1,2))),-1,1)
            assert abs(audit-mm).max()<1e-12
        delta=per[family+'__optimized']-per[family+'__previous'];boot=np.array([rng.choice(delta,TEST_N,replace=True).mean() for _ in range(5000)])
        dp=per[family+'__optimized__phase'].mean(-1)-per[family+'__previous__phase'].mean(-1);bp=np.array([rng.choice(dp,TEST_N,replace=True).mean() for _ in range(5000)])
        group['paired_delta']=dict(mean=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),phase_mean=float(dp.mean()),phase_ci95=np.quantile(bp,[.025,.975]).tolist());reports[family]=group;write(OUT/'test_progress_results.json',reports)
    np.savez_compressed(OUT/'per_example_test.npz',seeds=seeds,snr=rho,**per)
    result=dict(n=TEST_N,all_rows_retained=True,new_test_seeds=[TEST_SEED,TEST_SEED+TEST_N-1],primary_family='variable',results=reports,selection=frozen,observation='G*h+n',time_phase_alignment=False,full_population_improves_same_test=reports['variable']['paired_delta']['ci95'][1]<0,phase_improves_same_test=reports['variable']['paired_delta']['phase_ci95'][1]<0)
    write(OUT/'final_results.json',result);status(OUT,'completed',primary_mean=reports['variable']['optimized']['psd']['mean'],improved_same_test=result['full_population_improves_same_test'],phase_improves_same_test=result['phase_improves_same_test'])
if __name__=='__main__':main()
