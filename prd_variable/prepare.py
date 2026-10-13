import json,time,h5py,numpy as np,torch
from prd_physical.common import Preprocess,write,N
from prd_physical.prepare import PARENT,crop_meta
from prd_physical.physics import Simulator
from prd_frequency.features import features
from prd_lowrank.common import legacy_features,ram_guard,status
from .settings import *
from .data import variable_mask,fixed_mask
from .model import VariableDGF

def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'prepared.json').exists():return
    torch.set_num_threads(2);model=VariableDGF().eval();pre=Preprocess(np.load(CAL)['psd'],'white')
    sim=Simulator(); signs=np.random.default_rng(SEED).choice([-1.,1.],(VIEWS,TRAIN_N)).astype('f4');signs[0]=1
    (OUT/'cache').mkdir(exist_ok=True);np.save(OUT/'cache'/'train_sign.npy',signs)
    for split,count,num in [('validation',VAL_N,3),('train',TRAIN_N,VIEWS)]:
        h=np.load(CACHE/(split+'_truth.npy'),mmap_mode='r');seeds=np.load(CACHE/(split+'_seeds.npy'));pars=np.load(CACHE/(split+'_parameters.npy'));original=np.load(CACHE/(split+'_mask.npy'))
        with h5py.File(PARENT/(split+'.h5'),'r') as parent:
            for j in range(num):
                path=OUT/'cache'/f'{split}{j}';path.mkdir(exist_ok=True)
                mask=variable_mask(original,seeds,410000000+j*1000000) if split=='train' or j==0 else fixed_mask(original,66 if j==1 else 100)
                arrays={}
                for k,shape in FIELDS.items():
                    p=path/(k+'.npy'); arrays[k]=np.load(p,mmap_mode='r+') if p.exists() else np.lib.format.open_memmap(p,mode='w+',dtype='f4',shape=(count,)+shape)
                    assert arrays[k].shape==(count,)+shape
                progress=path/'progress.json';done=json.loads(progress.read_text())['completed'] if progress.exists() else 0
                t=time.monotonic();begin=done;last=0
                for lo in range(done,count,32):
                    hi=min(lo+32,count); noises=[]
                    for i in range(lo,hi):
                        start,old=crop_meta(pars[i],int(seeds[i]));assert np.array_equal(old,original[i])
                        if split=='train' and j==1:n=sim.noise(NOISE_SEED+i)[:,start:start+N]
                        else:n=parent['noise'][i,:,start:start+N]
                        noises.append(n.astype('f4'))
                    sign=signs[j,lo:hi] if split=='train' else np.ones(hi-lo,'f4')
                    physical=h[lo:hi]*(1-mask[lo:hi,None])+np.stack(noises)
                    u=pre.encode(physical)*sign[:,None,None]
                    q=legacy_features(u).numpy(); low=features(u)
                    with torch.inference_mode():bands=model.input_bands(torch.from_numpy(u)).numpy()
                    for k,v in dict(input=u,q=q,q_low=low,bands=bands,mask=mask[lo:hi]).items():
                        assert np.isfinite(v).all();arrays[k][lo:hi]=v
                    if time.monotonic()-last>30 or hi==count:
                        for a in arrays.values():a.flush()
                        elapsed=time.monotonic()-t;rate=(hi-begin)/max(elapsed,.001)
                        write(progress,dict(completed=hi,count=count))
                        status(OUT,'preparing_variable_gaps',split=split,view=j,completed=hi,total=count,rate=rate,view_remaining_seconds=(count-hi)/max(rate,.001))
                        ram_guard(OUT);last=time.monotonic()
                del arrays
    write(OUT/'prepared.json',dict(passed=True,train_unique=TRAIN_N,views=VIEWS,exposures_per_epoch=TRAIN_N*VIEWS,gap_samples=list(LENGTHS),gap_fraction=[min(LENGTHS)/N,max(LENGTHS)/N],validation_sets=['uniform_variable','fixed66','fixed100'],noise_seed_range=[NOISE_SEED,NOISE_SEED+TRAIN_N-1],observation='G*h+n',gap_centers_preserved=True,features_recomputed_from_observation=True,full_population=True,test_read=False))
if __name__=='__main__':main()
