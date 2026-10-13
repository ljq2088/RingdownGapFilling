import numpy as np,torch,psutil,time
from prd_physical.common import N
from .settings import *

def variable_mask(original,seeds,stream):
    original=np.asarray(original); a=original.argmax(-1); result=np.zeros_like(original)
    lengths=np.array([LENGTHS[int(np.random.default_rng(int(s)+stream).integers(len(LENGTHS)))] for s in seeds])
    start=a-(lengths-66)//2
    assert (start>=0).all() and (start+lengths<=N).all()
    for i,(s,l) in enumerate(zip(start,lengths)):result[i,s:s+l]=1
    return result

def fixed_mask(original,length):
    result=np.zeros_like(original); starts=original.argmax(-1)-(length-66)//2
    assert length%2==0 and (starts>=0).all() and (starts+length<=N).all()
    for i,s in enumerate(starts):result[i,s:s+length]=1
    return result

class Dataset:
    def __init__(self,split):
        self.split=split;self.views=[];self.ram=[];self.gpu=[]
        count=TRAIN_N if split=='train' else VAL_N
        for j in range(VIEWS if split=='train' else 3):
            path=OUT/'cache'/f'{split}{j}'
            self.views.append({k:torch.from_numpy(np.load(path/(k+'.npy'),mmap_mode='c')) for k in FIELDS})
            self.ram.append({});self.gpu.append({})
        self.target=torch.from_numpy(np.load(CACHE/(split+'_target.npy'),mmap_mode='c'))
        self.sign=torch.from_numpy(np.load(OUT/'cache'/'train_sign.npy')) if split=='train' else torch.ones((3,count))
        self.count=count
    def cache(self):
        # One complete training view uses ~12 GiB. Keep >=8 GiB available RAM.
        if self.split=='train':
            for k,v in self.views[0].items():
                required=v.numel()*v.element_size()
                if psutil.virtual_memory().available-required<8*2**30:break
                self.ram[0][k]=v.clone()
        free,total=torch.cuda.mem_get_info();budget=min(max(0,free-5*2**30),int(total*.78)-torch.cuda.memory_allocated()-3*2**30)
        prefix=self.count if self.split=='validation' else 16384
        for j in range(len(self.views)):
            n=prefix if self.split=='validation' else (prefix if j==0 else 0)
            required=sum(v[:n].numel()*v.element_size() for v in self.views[j].values())
            if n and required<budget:
                self.gpu[j]={k:v[:n].cuda() for k,v in self.views[j].items()};budget-=required
        return dict(ram_bytes=sum(v.numel()*v.element_size() for a in self.ram for v in a.values()),gpu_bytes=sum(v.numel()*v.element_size() for a in self.gpu for v in a.values()))
    def batch(self,indices):
        indices=np.asarray(indices);source=indices%self.count;view=indices//self.count
        result={k:torch.empty((len(indices),)+shape,device='cuda') for k,shape in FIELDS.items()}
        for j in np.unique(view):
            pos=np.flatnonzero(view==j);ix=source[pos];cache=self.gpu[j];n=len(cache['input']) if cache else 0
            for cached in (True,False):
                selected=(ix<n) if cached else (ix>=n)
                if not selected.any():continue
                output=torch.as_tensor(pos[selected],device='cuda');ii=torch.as_tensor(ix[selected],device='cuda' if cached else 'cpu')
                for k in FIELDS:
                    bank=cache[k] if cached else self.ram[j].get(k,self.views[j][k])
                    result[k].index_copy_(0,output,bank.index_select(0,ii).cuda())
        target=self.target.index_select(0,torch.as_tensor(source)).cuda()
        signs=self.sign[torch.as_tensor(view),torch.as_tensor(source)].cuda()
        result['target']=target*signs[:,None,None]
        return result
