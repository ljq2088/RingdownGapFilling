"""Generate a fresh physical population after full-budget validation selection freezes."""
import json, time
import numpy as np, torch
from prd_physical.common import N, FS, Preprocess, metric, summary, write, sha
from prd_physical.physics import Simulator
from prd_physical.prepare import crop_meta
from prd_frequency.model import FrequencyDGF
from prd_strategy.latent import CoefficientAdapter
from .settings import *
from .common import status, check_sources, ram_guard, load_backbone, inferred
from .data import packet, reconstruct, FIELDS

def main():
    torch.set_num_threads(2)
    check_sources();ram_guard(OUT)
    frozen=json.loads((OUT/'selection_frozen.json').read_text())
    assert frozen['new_test_read'] is False and frozen['full_training_sources']==TRAIN_N
    heads={}
    for chosen in frozen['candidates']:
        kind=chosen['variant'];path=OUT/kind/'best.pth'
        assert sha(path)==chosen['checkpoint_sha256']
        ck=torch.load(path,map_location='cpu',weights_only=True)
        model=CoefficientAdapter(RANK).eval();model.load_state_dict(ck['state_dict']);heads[kind]=model
    # The initial projection remains a predeclared structural reference.
    heads['fixed_projection']=CoefficientAdapter(RANK).eval()
    basis64=np.load(PROBE/'training_direction_basis.npz')['vectors'][:,:RANK]
    basis=torch.tensor(basis64,dtype=torch.float32)
    test=OUT/'fresh_test';test.mkdir(exist_ok=True)
    shapes={'truth':(TEST_N,2,N),'noise':(TEST_N,2,N),'mask':(TEST_N,N),'parameters':(TEST_N,15)}
    maps={}
    for k,shape in shapes.items():
        p=test/(k+'.npy')
        maps[k]=np.load(p,mmap_mode='r+') if p.exists() else np.lib.format.open_memmap(p,mode='w+',dtype='f8' if k=='parameters' else 'f4',shape=shape)
        assert maps[k].shape==shape
    progress=test/'progress.json';completed=json.loads(progress.read_text())['completed'] if progress.exists() else 0
    sim=Simulator();t=time.monotonic();begin=completed
    for i in range(completed,TEST_N):
        seed=TEST_SEED+i;h,n,par=sim.example(seed);start,m=crop_meta(par,seed)
        maps['truth'][i]=h[:,start:start+N];maps['noise'][i]=n[:,start:start+N];maps['mask'][i]=m;maps['parameters'][i]=par
        if (i+1)%50==0 or i+1==TEST_N:
            for a in maps.values():a.flush()
            write(progress,dict(completed=i+1,total=TEST_N))
            elapsed=time.monotonic()-t
            status(OUT,'generating_new_independent_test',completed=i+1,total=TEST_N,remaining_seconds=(TEST_N-i-1)*elapsed/max(i+1-begin,1))
            ram_guard(OUT)
    seeds=np.arange(TEST_SEED,TEST_SEED+TEST_N);np.save(test/'seeds.npy',seeds)
    for split in ['train','validation']:
        assert not np.isin(seeds,np.load(CACHE/(split+'_seeds.npy'))).any()
    assert (maps['mask'].sum(-1)==66).all()
    cal=np.load(CAL);pre=Preprocess(cal['psd'],'white')
    backbone=load_backbone();previous=FrequencyDGF('continuation').cpu().eval()
    paths={name:OUT/(name+'_test_prediction.npy') for name in ['previous_joint','frequency_snapshot',*heads]}
    predictions={name:np.lib.format.open_memmap(p,mode='w+',dtype='f4',shape=(TEST_N,2,N)) for name,p in paths.items()}
    t=time.monotonic();last=0.
    for lo in range(0,TEST_N,INFERENCE_BATCH):
        hi=min(lo+INFERENCE_BATCH,TEST_N)
        h=maps['truth'][lo:hi].astype(float);m=maps['mask'][lo:hi]
        u=pre.encode(h*(1-m[:,None])+maps['noise'][lo:hi])
        p=inferred(backbone,u,m);old=inferred(previous,u,m)
        predictions['frequency_snapshot'][lo:hi]=p;predictions['previous_joint'][lo:hi]=old
        values={k:torch.from_numpy(v) for k,v in packet(u,p,m,basis64,np.ones(hi-lo)).items()}
        with torch.inference_mode():
            for name,head in heads.items():predictions[name][lo:hi]=reconstruct(head,values,basis).numpy()
        if time.monotonic()-last>30 or hi==TEST_N:
            status(OUT,'evaluating_fresh_test',completed=hi,total=TEST_N,elapsed_seconds=time.monotonic()-t);last=time.monotonic()
    truth=maps['truth'].astype(float);hann=np.hanning(N);per={};reports={};errors={}
    for name,y in predictions.items():
        y.flush();p=pre.decode(y);mm=metric(truth,p,cal['psd']);tm=metric(truth,p);hm=metric(truth*hann,p*hann,cal['hann_psd']);l2=np.sqrt(((p-truth)**2).sum((1,2))/(truth*truth).sum((1,2)))
        reports[name]=dict(psd=summary(mm),time=summary(tm),hann_psd=summary(hm),relative_l2=summary(l2))
        for k,a in [('psd',mm),('time',tm),('hann_psd',hm),('relative_l2',l2)]:per[name+'__'+k]=a
        H=np.fft.rfft(truth)[...,1:-1]/np.sqrt(cal['psd'][:,1:-1]);P=np.fft.rfft(p)[...,1:-1]/np.sqrt(cal['psd'][:,1:-1])
        check=1-np.clip((H.conj()*P).real.sum((1,2))/np.sqrt((abs(H)**2).sum((1,2))*(abs(P)**2).sum((1,2))),-1,1)
        err=float(abs(check-mm).max());assert err<1e-12;errors[name]=err
    preferred=frozen['preferred'];rng=np.random.default_rng(SEED+100);paired={}
    for name in reports:
        if name==preferred:continue
        delta=per[preferred+'__psd']-per[name+'__psd']
        bootstrap=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(5000)])
        paired[name]=dict(mean_delta=float(delta.mean()),ci95=np.quantile(bootstrap,[.025,.975]).tolist())
    np.savez_compressed(OUT/'per_example_test.npz',seeds=seeds,parameters=maps['parameters'],**per)
    result=dict(preferred_from_validation=preferred,n=TEST_N,all_rows_retained=True,new_independent_test=True,test_seeds=[TEST_SEED,TEST_SEED+TEST_N-1],noise_augmentation_seeds=[NOISE_SEED,NOISE_SEED+(VIEWS-1)*TRAIN_N-1],observation='G*h+n',time_phase_maximized=False,results=reports,paired=paired,selection=frozen,improved_vs_frequency_snapshot_on_same_new_test=paired['frequency_snapshot']['ci95'][1]<0,historical_numeric_target=.005368768966470572,historical_comparison_scope='different population; contextual numerical target',uncertainty_scope='paired source bootstrap conditional on frozen checkpoints; training-seed uncertainty separate')
    write(OUT/'final_metric_audit.json',dict(passed=True,independent_psd_max_errors=errors,all_rows_retained=True,checkpoint_hashes_verified=True,train_validation_test_seed_ranges_disjoint=True))
    write(OUT/'final_results.json',result);status(OUT,'completed',preferred=preferred,psd_mean=reports[preferred]['psd']['mean'],improved_same_test=result['improved_vs_frequency_snapshot_on_same_new_test'])

if __name__=='__main__':main()
