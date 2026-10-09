"""Cache fixed observed-input db4 bands; no target or learned weight is used."""
import time
from pathlib import Path
import torch,h5py
from .common import sha,write,N
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'FillingGap/results_prd_physical_transfer_20261009'

def attach_bands(dataset,kind,split,model):
 path=OUT/(split+'_'+kind+'_db4_bands.h5')
 provenance={name:sha(ROOT/name) for name in ['utils/wavelet.py','utils/normalize.py','config/config.py']}
 input_sha=sha(OUT/(split+'_'+kind+'_features.h5'))
 cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
 if not path.exists():
  tmp=path.with_suffix('.tmp');t=time.monotonic();count=len(dataset['input'])
  with h5py.File(tmp,'w') as f:
   d=f.create_dataset('bands',shape=(count,2,8,N),dtype='f4',chunks=(16,2,8,N))
   with torch.no_grad():
    for lo in range(0,count,128):
     u=dataset['input'][lo:lo+128].cuda();d[lo:lo+len(u)]=model.input_bands(u).cpu().numpy()
   f.attrs['completed']=count;f.attrs['input_features_sha256']=input_sha
   for name,value in provenance.items():f.attrs[name]=value
   f.attrs['seconds']=time.monotonic()-t
  tmp.replace(path)
 with h5py.File(path,'r') as f:
  assert f.attrs['completed']==len(dataset['input']) and f.attrs['input_features_sha256']==input_sha
  for name,value in provenance.items():assert f.attrs[name]==value
  dataset['bands']=torch.from_numpy(f['bands'][:])
 assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state())
 return dict(path=str(path),bytes=dataset['bands'].numel()*dataset['bands'].element_size(),provenance=provenance,input_features_sha256=input_sha,RNG_unchanged=True)
