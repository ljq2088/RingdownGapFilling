import gc,numpy as np,torch
from .settings import OUT,CACHE,TRAIN_N,VAL_N

class Data:
    def __init__(self,split,variant):
        self.count=TRAIN_N if split=='train' else VAL_N;self.variant=variant
        names=['prediction','features','mask','peak','scale','sign']+(['obs'] if variant=='tail_unet' else [])
        self.banks=[{k:torch.from_numpy(np.load(OUT/'cache'/f'{split}{j}'/(k+'.npy'),mmap_mode='c')) for k in names} for j in range(4 if split=='train' else 1)]
        self.target=torch.from_numpy(np.load(CACHE/(split+'_target.npy'),mmap_mode='c'))
        pars=np.load(CACHE/(split+'_parameters.npy'));self.parameters=torch.tensor(np.c_[np.log(pars[:,9]),pars[:,10]],dtype=torch.float32)
        self.gpu=[{} for _ in self.banks];self.target_gpu=None
    def cache(self):
        # Fixed cache budget leaves capacity for model activations and the desktop.
        budget=max(0,min(5.2*2**30,torch.cuda.mem_get_info()[0]-4.2*2**30-2.5*2**30))
        spent=0
        for j,bank in enumerate(self.banks):
            for k in ['features','mask','peak','scale','sign','prediction','obs']:
                if k not in bank:continue
                v=bank[k];size=v.numel()*v.element_size()
                if spent+size>budget:continue
                self.gpu[j][k]=v.cuda();spent+=size
        size=self.target.numel()*self.target.element_size()
        if spent+size<budget:self.target_gpu=self.target.cuda();spent+=size
        self.parameters=self.parameters.cuda()
        return dict(gpu_bytes=spent,fields=[list(x) for x in self.gpu])
    def batch(self,indices):
        ix=np.asarray(indices);source=ix%self.count;view=ix//self.count;v={}
        for k,x in self.banks[0].items():
            result=torch.empty((len(ix),)+x.shape[1:],device='cuda',dtype=x.dtype)
            for j in np.unique(view):
                pos=np.flatnonzero(view==j);src=source[pos];bank=self.gpu[j].get(k,self.banks[j][k]);ii=torch.as_tensor(src,device=bank.device)
                result.index_copy_(0,torch.as_tensor(pos,device='cuda'),bank.index_select(0,ii).cuda())
            v[k]=result
        tar=self.target_gpu if self.target_gpu is not None else self.target
        v['target']=tar.index_select(0,torch.as_tensor(source,device=tar.device)).cuda()*v['sign'][:,None,None]
        v['parameters']=self.parameters.index_select(0,torch.as_tensor(source,device=self.parameters.device)).cuda();v['mask']=v['mask'].float()
        return v
