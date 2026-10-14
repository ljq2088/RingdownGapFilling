import psutil,numpy as np,torch
from prd_conditioned.data import MixedData
from prd_variable.settings import FIELDS
from .settings import OUT

class Data(MixedData):
    def __init__(self,split,variant):
        super().__init__(split);self.variant=variant
        if split=='train' and variant=='noise_mix':
            self.views[2]={k:torch.from_numpy(np.load(OUT/'mixed_cache'/(k+'.npy'),mmap_mode='c')) for k in FIELDS}
            self.sign[2]=1.
    def cache(self,pilot=False):
        # Materialize only pilot prefixes; preserve RAM and VRAM for the desktop.
        if self.split=='train' and not pilot:
            for j,view in enumerate(self.views):
                for k,v in view.items():
                    if psutil.virtual_memory().available-v.numel()*v.element_size()<9*2**30:continue
                    self.ram[j][k]=v.clone()
        free,total=torch.cuda.mem_get_info()
        # The measured worst-case batch requires ~4.9 GiB, including Adam.
        # Leave that activation budget plus 4 GiB for external GPU use.
        activation_reserve=int(5.5*2**30)
        budget=min(max(0,free-4*2**30-activation_reserve),int(total*.78)-torch.cuda.memory_allocated()-activation_reserve)
        sizes=[sum(v.numel()*v.element_size()/len(v) for v in view.values()) for view in self.views]
        desired=self.count if self.split=='validation' else (6000 if pilot else 8192)
        n=min(desired,int(budget/max(sum(sizes),1)))
        n=(n//128)*128
        if self.split=='validation' and budget>=sum(sizes)*self.count:n=self.count
        if n>0:
            for j,view in enumerate(self.views):self.gpu[j]={k:v[:n].cuda() for k,v in view.items()}
        return dict(ram_bytes=sum(v.numel()*v.element_size() for a in self.ram for v in a.values()),
                    gpu_prefix=n,gpu_bytes=sum(v.numel()*v.element_size() for a in self.gpu for v in a.values()))
