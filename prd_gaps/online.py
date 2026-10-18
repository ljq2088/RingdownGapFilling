"""Deterministic epoch-wise augmentation, with no source parameters as model inputs."""
import hashlib, json, multiprocessing
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from torch.utils.data import Dataset
from .preprocess import make_features
from .masks import erode_valid

def event_masks(events, config):
    """Exact event/stencil intersection matching propagate_tdi on the padded grid."""
    n,fs,r=config['n'],config['fs'],config['order']+2
    xyz=np.ones((3,n),bool)
    for c in range(3):
        for term in config['tdi_combinations']:
            i,j=divmod(term['link'],10); link=10*((i-1+c)%3+1)+((j-1+c)%3+1)
            delay=len(term['links_for_delay'])*config['arm_seconds']*fs
            for event in events:
                if event['link'] not in ('all',link): continue
                a=int(np.floor(event['start_s']*fs));b=int(np.ceil((event['start_s']+event['duration_s'])*fs))
                lo=max(0,int(np.floor(a-r+delay))-1);hi=min(n,int(np.ceil(b+r+delay))+1)
                if hi<=lo:continue
                t=np.arange(lo,hi);q=t-delay
                bad=(np.floor(q)-r<=b-1)&(np.ceil(q)+r>=a)
                xyz[c,t[bad]]=False
    return np.stack((xyz[2]&xyz[0],xyz.all(axis=0)))

def make_events(meta,config,rng,epoch,training):
    # Frozen validation: 90% peak-overlapping design, 10% uniform random control.
    targeted=meta['source_index']%10!=9
    fraction=min(max(epoch,0)/15.,1.) if training else 1.
    low=10+(33-10)*fraction;high=33+(53-33)*fraction
    duration=float(rng.uniform(low,high))
    if training and epoch>=15 and rng.random()<.1:duration=float(rng.uniform(100,200))
    peak=meta['detector_energy_peak_index']/config['fs']
    if targeted:
        start=peak-rng.uniform(.2,.8)*duration-rng.uniform(0.,4*config['arm_seconds'])
    else:
        start=rng.uniform(0.,config['n']/config['fs']-duration-200)
    start=float(np.clip(start,0,config['n']/config['fs']-duration-200))
    link='all' if meta['source_index']%4==0 else int(rng.choice(config['links']))
    events=[dict(link=link,start_s=start,duration_s=duration)]
    if meta['source_index']%8==3:
        events.append(dict(link=int(rng.choice(config['links'])),start_s=start+150.,duration_s=1.))
    return events

class SourceBank(Dataset):
    def __init__(self,root,split,limit=None):
        self.root=Path(root);self.config=json.loads((self.root/'manifest.json').read_text())
        self.split=split;self.training=split=='train';self.epoch=multiprocessing.Value('i',0)
        if split not in ('train','val'):raise ValueError('Training loader is restricted to development splits')
        self.files=sorted((self.root/split).glob('*.npz'))
        if limit:self.files=self.files[:limit]
        if not self.files:raise ValueError('No source records available')
        self.verified=set();self.hashes={}
        if self.config.get('complete'):
            content=(self.root/'index.jsonl').read_bytes()
            if hashlib.sha256(content).hexdigest()!=self.config['index_sha256']:raise ValueError('Bank index checksum mismatch')
            entries=[json.loads(line) for line in content.splitlines()]
            self.hashes={e['file']:e['sha256'] for e in entries}
            seeds=[e['seed'] for e in entries]
            if len(set(seeds))!=len(seeds):raise ValueError('Overlapping source seeds')
            if len(list((self.root/split).glob('*.npz')))!=self.config[split]:raise ValueError('Incomplete source split')
        with np.load(self.root/'calibration.npz') as a:
            self.psd=a['psd']/a['scale'][:,None]**2;self.kernel=a['kernel'];self.scale=a['scale'];self.parent_psd=a['parent_psd']

    def __len__(self):return len(self.files)
    def set_epoch(self,epoch):self.epoch.value=int(epoch)

    @lru_cache(maxsize=64)
    def read_source(self,index):
        path=self.files[index]
        if self.hashes and index not in self.verified:
            if hashlib.sha256(path.read_bytes()).hexdigest()!=self.hashes[str(path.relative_to(self.root))]:raise ValueError('Source checksum mismatch')
            self.verified.add(index)
        with np.load(path) as a:return a['target'],json.loads(str(a['metadata']))

    def __getitem__(self,index):
        target,meta=self.read_source(index)
        epoch=self.epoch.value if self.training else 0
        rng=np.random.default_rng(np.random.SeedSequence([meta['source_seed'],epoch,9173]))
        n=self.config['parent_n'];fs=self.config['fs']
        z=(rng.normal(size=self.parent_psd.shape)+1j*rng.normal(size=self.parent_psd.shape))*np.sqrt(n*fs*self.parent_psd/4)
        z[:,0]=0;z[:,-1]=rng.normal(size=2)*np.sqrt(n*fs*self.parent_psd[:,-1]/2)
        pad=self.config['pad'];noise=np.fft.irfft(z,n=n)[:,pad:pad+self.config['n']]
        events=make_events(meta,self.config,rng,epoch,self.training)
        valid=event_masks(events,self.config)
        if self.training and epoch>=15 and rng.random()<.1:
            guard=int(np.ceil(7*self.config['arm_seconds']*fs))+self.config['order']+2
            valid=erode_valid(np.pad(valid,((0,0),(guard,guard)),constant_values=True),guard)[:,guard:-guard]
        features=make_features(target*self.scale[:,None]+noise,valid,self.kernel,self.scale)
        start=meta['detector_energy_peak_index'];stop=min(self.config['n'],start+int(np.ceil(6*meta['tau220_estimate_seconds']*fs)))
        stop-=(stop-start)%2
        roi=target[:,start:stop].astype(float)**2
        removed=float((roi*(~valid[:,start:stop])).sum()/max(roi.sum(),1e-30))
        info=np.array([start,stop,float((~valid[:,start:stop]).any()),removed],np.float64)
        return torch.from_numpy(features),torch.from_numpy(target),torch.from_numpy(valid.astype(np.float32)),torch.from_numpy(info),index
