import json,time,numpy as np,torch
from prd_physical.common import Preprocess,write,N
from prd_physical.prepare import crop_meta
from prd_physical.physics import Simulator
from prd_lowrank.common import legacy_features,status
from prd_frequency.features import features
from prd_variable.model import VariableDGF
from .settings import *
from .common import check,wait_resources

def main():
    check();torch.set_num_threads(2);wait_resources()
    path=OUT/'fixed_noise_cache';path.mkdir(exist_ok=True)
    if (path/'complete.json').exists():return
    count=TRAIN_N;h=np.load(CACHE/'train_truth.npy',mmap_mode='r');seeds=np.load(CACHE/'train_seeds.npy');pars=np.load(CACHE/'train_parameters.npy');mask=np.load(CACHE/'train_mask.npy')
    sign=np.random.default_rng(SEED+7).choice([-1.,1.],count).astype('f4');np.save(path/'sign.npy',sign)
    arrays={k:np.load(path/(k+'.npy'),mmap_mode='r+') if (path/(k+'.npy')).exists() else np.lib.format.open_memmap(path/(k+'.npy'),mode='w+',dtype='f4',shape=(count,)+shape) for k,shape in FIELDS.items()}
    progress=path/'progress.json';done=json.loads(progress.read_text())['completed'] if progress.exists() else 0
    sim=Simulator();model=VariableDGF().eval();pre=Preprocess(np.load(CAL)['psd'],'white');t=time.monotonic();begin=done;last=0
    for lo in range(done,count,32):
        hi=min(lo+32,count);noise=[]
        for i in range(lo,hi):
            start,m=crop_meta(pars[i],int(seeds[i]));assert np.array_equal(m,mask[i]);noise.append(sim.noise(2158200000+i)[:,start:start+N])
        u=pre.encode(h[lo:hi]*(1-mask[lo:hi,None])+np.stack(noise))*sign[lo:hi,None,None]
        q=legacy_features(u).numpy();low=features(u)
        with torch.inference_mode():bands=model.input_bands(torch.from_numpy(u)).numpy()
        for k,v in dict(input=u,q=q,q_low=low,bands=bands,mask=mask[lo:hi]).items():assert np.isfinite(v).all();arrays[k][lo:hi]=v
        if time.monotonic()-last>30 or hi==count:
            for a in arrays.values():a.flush()
            write(progress,dict(completed=hi,total=count));elapsed=time.monotonic()-t
            status(OUT,'preparing_fixed_gap_new_noise',completed=hi,total=count,remaining_seconds=(count-hi)*elapsed/max(hi-begin,1));last=time.monotonic();wait_resources()
    write(path/'complete.json',dict(passed=True,noise_seed_range=[2158200000,2158259999],gap_samples=66,global_sign_augmentation=True,features_recomputed=True))
if __name__=='__main__':main()
