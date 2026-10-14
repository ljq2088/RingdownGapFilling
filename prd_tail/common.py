import json,time,psutil,torch
from prd_physical.common import sha,write
from prd_lowrank.common import status
from .settings import ROOT,OUT

def configure():
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.70)

def guard():
    if (OUT/'STOP').exists():raise RuntimeError('Requested stop at checkpoint boundary')
    while psutil.virtual_memory().available<5*2**30:
        status(OUT,'waiting_for_ram',available_gib=psutil.virtual_memory().available/2**30);time.sleep(10)

def check():
    for name,expected in json.loads((OUT/'frozen_manifest.json').read_text()).items():
        if sha(ROOT/name)!=expected:raise RuntimeError('Frozen dependency changed: '+name)

def analytic(x):
    n=x.shape[-1];a=torch.zeros(n,device=x.device,dtype=x.dtype);a[0]=a[n//2]=1;a[1:n//2]=2
    return torch.fft.ifft(torch.fft.fft(x)*a)
