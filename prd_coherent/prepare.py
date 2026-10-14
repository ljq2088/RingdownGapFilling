import json,time,numpy as np,torch
from prd_physical.common import N,Preprocess,write,sha
from prd_lowrank.common import legacy_features,status
from prd_frequency.features import features
from prd_variable.data import variable_mask
from prd_variable.settings import FIELDS
from prd_conditioned.data import MixedData
from .settings import OUT,CAL,PROBE,CACHE,TRAIN_N,SEED
from .model import GapOperator

def gram():
    if (OUT/'mask_gram_complete.json').exists():return
    B=np.load(PROBE/'training_direction_basis.npz')['vectors'][:,:128]
    b=torch.tensor(B.T.reshape(128,2,N),device='cuda',dtype=torch.float32)
    op=GapOperator().cuda();path=OUT/'mask_gram.npy'
    result=np.lib.format.open_memmap(path,mode='w+',dtype='f4',shape=(132*21,128,128))
    with torch.inference_mode():
        for lo in range(0,len(result),8):
            hi=min(lo+8,len(result));mask=np.zeros((hi-lo,N),dtype='f4')
            for j,index in enumerate(range(lo,hi)):
                original=132+index//21;length=64+2*(index%21);start=original-(length-66)//2
                mask[j,start:start+length]=1
            f=op(b[None].expand(hi-lo,-1,-1,-1),torch.from_numpy(mask).cuda()).flatten(2)
            h=f@f.transpose(1,2);result[lo:hi]=h.cpu().numpy()
            if lo%256==0:status(OUT,'preparing_gap_operator',completed=hi,total=len(result))
    result.flush();write(OUT/'mask_gram_complete.json',dict(geometries=len(result),rank=128,sha256=sha(path),operator='P0 W G W^-1',basis_sha=sha(PROBE/'training_direction_basis.npz')))

def mixed(count,generation=0):
    # Feature preparation preserves the training RNG, including after restart.
    with torch.random.fork_rng(devices=[0]):return _mixed(count,generation)

def _mixed(count,generation=0):
    path=OUT/'mixed_cache';path.mkdir(exist_ok=True);progress=path/'progress.json'
    prev=json.loads(progress.read_text()) if progress.exists() else {}
    done=prev.get('completed',0) if prev.get('generation')==generation else 0
    if done>=count:return
    arrays={k:np.load(path/(k+'.npy'),mmap_mode='r+') if (path/(k+'.npy')).exists() else np.lib.format.open_memmap(path/(k+'.npy'),mode='w+',dtype='f4',shape=(TRAIN_N,)+shape) for k,shape in FIELDS.items()}
    data=MixedData('train');op=GapOperator().cuda()
    # Only preprocessing is needed; this object also fixes db4 conventions.
    from prd_conditioned.model import ConditionedDGF
    network=ConditionedDGF('noise_phase').eval()
    original=np.load(CACHE/'train_mask.npy',mmap_mode='r');seeds=np.load(CACHE/'train_seeds.npy')
    tick=time.monotonic();last=0;begin=done
    for lo in range(done,count,32):
        hi=min(lo+32,count);h=data.target[lo:hi].cuda()
        with torch.inference_mode():
            residual=[]
            for j in (0,1):
                m=data.views[j]['mask'][lo:hi].cuda();u=data.views[j]['input'][lo:hi].cuda()
                sign=data.sign[j,lo:hi].cuda()
                residual.append(u-op.physical_mask(h*sign[:,None,None],m))
            angles=np.array([np.random.default_rng(int(s)+SEED+generation*1000003).uniform(0,2*np.pi) for s in seeds[lo:hi]],dtype='f4')
            theta=torch.from_numpy(angles).cuda()[:,None,None]
            noise=theta.cos()*residual[0]+theta.sin()*residual[1]
            m=variable_mask(original[lo:hi],seeds[lo:hi],421500000+generation*1000000)
            u=(op.physical_mask(h,torch.from_numpy(m).cuda())+noise).cpu().numpy()
            bands=network.input_bands(torch.from_numpy(u)).numpy()
        q=legacy_features(u).numpy();low=features(u)
        for k,v in dict(input=u,mask=m,q=q,q_low=low,bands=bands).items():
            assert np.isfinite(v).all();arrays[k][lo:hi]=v
        if time.monotonic()-last>30 or hi==count:
            for a in arrays.values():a.flush()
            elapsed=time.monotonic()-tick
            write(progress,dict(completed=hi,total=TRAIN_N,generation=generation,stream=421500000+generation*1000000,formula='cos(theta)*n0 + sin(theta)*n1',features_recomputed=True))
            status(OUT,'preparing_covariance_preserving_augmentation',completed=hi,target=count,generation=generation,remaining_seconds=(count-hi)*elapsed/max(hi-begin,1));last=time.monotonic()

def snr():
    path=OUT/'training_snr.npy'
    if path.exists():return
    y=np.load(CACHE/'train_target.npy',mmap_mode='r');rho=[]
    for lo in range(0,TRAIN_N,512):
        f=np.fft.rfft(y[lo:lo+512].astype(float))[...,1:-1]
        rho.extend(np.sqrt(2/N*(abs(f)**2).sum((1,2))))
    np.save(path,np.asarray(rho));write(OUT/'training_snr_summary.json',dict(n=len(rho),minimum=float(min(rho)),maximum=float(max(rho)),below20=int((np.asarray(rho)<20).sum())))
