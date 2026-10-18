"""Resumable, source-disjoint IMR bank; noise and gaps are generated at training time."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
import argparse, concurrent.futures, hashlib, json, time
from pathlib import Path
import numpy as np

_sim = None
_destination = None
_scale = None

def initialize(destination):
    global _sim, _destination, _scale
    from .physics import Simulator
    _destination = Path(destination); _sim = Simulator()
    with np.load(_destination/'calibration.npz') as a: _scale = a['scale']

def generate_one(job):
    from .physics import draw_parameters
    split, index, seed = job
    path = _destination/split/f'{index:06d}.npz'
    if path.exists():
        with np.load(path) as a:
            meta = json.loads(str(a['metadata']))
            if meta['source_seed'] != seed or a['target'].shape != (2,_sim.n):
                raise ValueError('Source bank file identity mismatch')
    else:
        rng = np.random.default_rng(seed+80000000)
        _sim.peak_seconds = float(rng.uniform(1024.,3072.))
        signal, _, meta = _sim.signal(draw_parameters(seed))
        meta.update(source_seed=seed, split=split, source_index=index)
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as f:
            np.savez_compressed(f,target=(signal/_scale[:,None]).astype(np.float32),metadata=json.dumps(meta))
        os.replace(temporary,path)
    return dict(split=split,index=index,seed=seed,file=str(path.relative_to(_destination)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def run(args):
    from .physics import Simulator, ARM_SECONDS
    from .preprocess import design_fir
    root = Path(args.output); root.mkdir(parents=True,exist_ok=True)
    config = dict(schema=3,train=args.train,val=args.val,n=32768,fs=2.,
                  seeds=dict(train=61000000,val=71000000),observation='G*(s+n)',
                  calibration_count=512,phase_reference='0.02/M_detector_seconds',
                  source_mass=[1e5,5e6],q=[.3,1.],redshift=[1.,6.],split='source-disjoint',
                  complete=False)
    manifest_path = root/'manifest.json'
    if manifest_path.exists():
        existing=json.loads(manifest_path.read_text())
        for key in ['schema','train','val','n','fs','seeds','phase_reference']:
            if existing[key]!=config[key]: raise ValueError(f'Resume configuration differs: {key}')
        config=existing
    else:
        sim=Simulator(); psd=sim.calibrate_psd(config['calibration_count'])
        kernel,scale=design_fir(psd,sim.fs,radius=32)
        np.savez(root/'calibration.npz',psd=psd,kernel=kernel,scale=scale,parent_psd=sim.psd)
        config.update(pad=sim.pad,parent_n=sim.parent_n,links=sim.links.tolist(),
                      tdi_combinations=sim.combinations,arm_seconds=ARM_SECONDS,order=sim.order,
                      source_code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in Path(__file__).parent.glob('*.py')})
        manifest_path.write_text(json.dumps(config,indent=2)+'\n')
    for split in ['train','val']: (root/split).mkdir(exist_ok=True)
    # Validation first: a frozen held-out development population is available
    # for early debugging while the larger training bank is being assembled.
    jobs=[(split,j,config['seeds'][split]+j) for split in ['val','train'] for j in range(config[split])]
    entries=[]; tic=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(args.workers,initializer=initialize,initargs=(str(root),)) as pool:
        with (root/'index.jsonl').open('w') as journal:
            for count,entry in enumerate(pool.map(generate_one,jobs,chunksize=4),1):
                entries.append(entry);journal.write(json.dumps(entry)+'\n');journal.flush()
                if count%32==0 or count==len(jobs):
                    elapsed=time.perf_counter()-tic
                    status=dict(stage='generation',completed=count,total=len(jobs),seconds=elapsed,
                                sources_per_second=count/elapsed,eta_seconds=(len(jobs)-count)*elapsed/count)
                    tmp=root/'generation_status.tmp';tmp.write_text(json.dumps(status)+'\n');os.replace(tmp,root/'generation_status.json')
                    print(json.dumps(status),flush=True)
    config['complete']=True
    config['index_sha256']=hashlib.sha256((root/'index.jsonl').read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(config,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--train',type=int,default=8192);p.add_argument('--val',type=int,default=512)
    p.add_argument('--workers',type=int,default=4)
    run(p.parse_args())
