import json,time,psutil,torch
from prd_physical.common import sha
from prd_lowrank.common import status
from prd_conditioned.common import check as check_previous
from .settings import ROOT,OUT

def check():
    check_previous()
    for filename in ('runtime_source_manifest.json','asset_manifest.json'):
        for name,expected in json.loads((OUT/filename).read_text()).items():
            if sha(ROOT/name)!=expected:raise RuntimeError('frozen file changed: '+name)

def resources():
    while psutil.virtual_memory().available<6*2**30:
        status(OUT,'waiting_for_ram',available_gib=psutil.virtual_memory().available/2**30);time.sleep(10)
    if (OUT/'STOP').exists():raise RuntimeError('STOP requested; resume from last completed epoch after removing STOP')

def configure():
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.78)
