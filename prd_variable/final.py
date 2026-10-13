import json,time,numpy as np,torch
from prd_physical.common import N,FS,Preprocess,metric,summary,sha,write
from prd_physical.physics import Simulator
from prd_physical.prepare import crop_meta
from prd_lowrank.common import legacy_features,status
from prd_frequency.features import features
from .settings import *
from .model import VariableDGF
from .data import variable_mask,fixed_mask
from .common import check,resource_guard

def main():
    check();resource_guard();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.78)
    frozen=json.loads((OUT/'selection_frozen.json').read_text());selected=frozen['selection'];ckpath=OUT/'joint_finetune/best.pth';assert sha(ckpath)==selected['checkpoint_sha256']
    test=OUT/'fresh_test';test.mkdir(exist_ok=True)
    shapes=dict(truth=(TEST_N,2,N),noise=(TEST_N,2,N),mask=(TEST_N,N),parameters=(TEST_N,15))
    maps={}
    for k,shape in shapes.items():
        p=test/(k+'.npy');maps[k]=np.load(p,mmap_mode='r+') if p.exists() else np.lib.format.open_memmap(p,mode='w+',dtype='f8' if k=='parameters' else 'f4',shape=shape)
    progress=test/'progress.json';completed=json.loads(progress.read_text())['completed'] if progress.exists() else 0
    sim=Simulator();t=time.monotonic();begin=completed
    for i in range(completed,TEST_N):
        seed=TEST_SEED+i;h,n,pars=sim.example(seed);start,m=crop_meta(pars,seed)
        maps['truth'][i]=h[:,start:start+N];maps['noise'][i]=n[:,start:start+N];maps['parameters'][i]=pars;maps['mask'][i]=m
        if (i+1)%100==0 or i+1==TEST_N:
            for a in maps.values():a.flush()
            write(progress,dict(completed=i+1,total=TEST_N));elapsed=time.monotonic()-t
            status(OUT,'generating_new_independent_test',completed=i+1,total=TEST_N,remaining_seconds=(TEST_N-i-1)*elapsed/max(i+1-begin,1));resource_guard()
    seeds=np.arange(TEST_SEED,TEST_SEED+TEST_N);np.save(test/'seeds.npy',seeds)
    for split in ('train','validation'):assert not np.isin(seeds,np.load(CACHE/(split+'_seeds.npy'))).any()
    baseline=VariableDGF().cuda().eval();model=VariableDGF().cuda().eval();model.load_state_dict(torch.load(ckpath,map_location='cuda',weights_only=True)['state_dict'])
    cal=np.load(CAL);pre=Preprocess(cal['psd'],'white');h=maps['truth'].astype(float)
    H=np.fft.rfft(h);snr=np.sqrt((4/(FS*N)*(abs(H[...,1:-1])**2/cal['psd'][:,1:-1])).sum((1,2)))
    reports={};per={};rng=np.random.default_rng(SEED+100)
    for family,m in [('variable',variable_mask(maps['mask'],seeds,410000000)),('fixed66',fixed_mask(maps['mask'],66)),('fixed100',fixed_mask(maps['mask'],100))]:
        predictions={name:np.lib.format.open_memmap(OUT/(family+'_'+name+'_prediction.npy'),mode='w+',dtype='f4',shape=(TEST_N,2,N)) for name in ('previous','optimized')}
        for lo in range(0,TEST_N,BATCH):
            hi=min(lo+BATCH,TEST_N);u=pre.encode(maps['truth'][lo:hi]*(1-m[lo:hi,None])+maps['noise'][lo:hi]);q=legacy_features(u).cuda();low=torch.from_numpy(features(u)).cuda();x=torch.from_numpy(u).cuda();mask=torch.from_numpy(m[lo:hi]).cuda()
            with torch.inference_mode():
                bands=model.input_bands(x)
                for name,network in [('previous',baseline),('optimized',model)]:predictions[name][lo:hi]=network(x,q,mask,bands,low).cpu().numpy()
            if lo%1024==0:status(OUT,'evaluating_test',family=family,completed=hi,total=TEST_N)
        group={}
        for name,pred in predictions.items():
            pred.flush();p=pre.decode(pred);mm=metric(h,p,cal['psd']);per[family+'__'+name]=mm
            A=np.fft.rfft(h)[...,1:-1]/np.sqrt(cal['psd'][:,1:-1]);B=np.fft.rfft(p)[...,1:-1]/np.sqrt(cal['psd'][:,1:-1]);cross=(A.conj()*B).real.sum((1,2));den=np.sqrt((abs(A)**2).sum((1,2))*(abs(B)**2).sum((1,2)));audit=1-np.clip(np.divide(cross,den,out=np.zeros_like(cross),where=den>0),-1,1);assert abs(audit-mm).max()<1e-12
            group[name]=dict(psd=summary(mm),time=summary(metric(h,p)),snr_ge10=summary(mm[snr>=10]),independent_psd_audit_max_error=float(abs(audit-mm).max()))
        delta=per[family+'__optimized']-per[family+'__previous'];boot=np.array([rng.choice(delta,TEST_N,replace=True).mean() for _ in range(5000)])
        group['paired_delta']=dict(mean=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist());reports[family]=group
        write(OUT/'test_progress_results.json',reports)
    np.savez_compressed(OUT/'per_example_test.npz',seeds=seeds,snr=snr,**per)
    result=dict(passed=True,n=TEST_N,all_rows_retained=True,new_test_seeds=[TEST_SEED,TEST_SEED+TEST_N-1],primary_family='variable',results=reports,selected=frozen,gap_samples=list(LENGTHS),observation='G*h+n',time_phase_alignment=False,full_population_improves_same_test=reports['variable']['paired_delta']['ci95'][1]<0,historical_numeric_target=.005368768966470572,historical_scope='different parameter distribution; compare same-test previous model separately')
    write(OUT/'final_results.json',result);status(OUT,'completed',primary_mean=reports['variable']['optimized']['psd']['mean'],improved_same_test=result['full_population_improves_same_test'])
if __name__=='__main__':main()
