import numpy as np,torch
from prd_variable.data import Dataset
from .settings import *
class MixedData(Dataset):
    def __init__(self,split):
        super().__init__(split)
        if split=='train':
            self.views.append({k:torch.from_numpy(np.load(OUT/'fixed_noise_cache'/(k+'.npy'),mmap_mode='c')) for k in FIELDS})
            self.ram.append({});self.gpu.append({});self.sign=torch.cat((self.sign,torch.from_numpy(np.load(OUT/'fixed_noise_cache/sign.npy'))[None,:]),dim=0)
    def batch(self,indices):
        v=super().batch(indices);ix=np.asarray(indices)
        v['sign']=self.sign[torch.as_tensor(ix//self.count),torch.as_tensor(ix%self.count)].cuda()
        return v

class PacketData:
    def __init__(self,split):
        self.count=TRAIN_N if split=='train' else VAL_N
        self.banks=[{k:torch.from_numpy(np.load(OUT/'packets'/f'{split}{j}'/(k+'.npy'))).cuda() for k in PACKET_FIELDS} for j in range(3)]
        self.target=torch.from_numpy(np.load(CACHE/(split+'_target.npy'))).cuda()
    def batch(self,indices):
        ix=torch.as_tensor(indices,device='cuda');source=ix.remainder(self.count);views=ix.div(self.count,rounding_mode='floor')
        v={k:torch.empty((len(ix),)+shape,device='cuda') for k,shape in PACKET_FIELDS.items()}
        for j in range(3):
            pos=torch.where(views==j)[0]
            if len(pos):
                for k in v:v[k].index_copy_(0,pos,self.banks[j][k][source[pos]])
        v['target']=self.target[source]*v['sign'][:,None,None]
        t=torch.arange(1056,device='cuda')[None,:];a=(v['start']*1056).round().long()[:,None];b=(v['length']*1056).round().long()[:,None]
        v['mask']=((t>=a)&(t<a+b)).float()
        return v
