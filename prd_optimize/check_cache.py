"""Independent reconstruction of selected expanded TRAIN rows; no test access."""
import numpy as np,torch,h5py
from prd_physical.common import Preprocess,sha,write
from prd_physical.prepare import crop_meta
from model.QTranTimeMixerMod import QTransformModule
from .settings import *
from .model import OptimizedDGF
from .data import load_maps

def main():
 torch.set_num_threads(2);a=load_maps("train");cal=np.load(OLD/"calibration_psd.npz");pre=Preprocess(cal["psd"],"white");q=QTransformModule();model=OptimizedDGF("continuation").cuda().eval();checked=[]
 with h5py.File(PARENT/"train.h5","r") as f:
  for i in [0,11999,12000,23456,40000,59999]:
   par=f["parameters"][i];seed=int(f["seeds"][i]);s,m=crop_meta(par,seed)
   h=f["truth"][i,:,s:s+1056];noise=f["noise"][i,:,s:s+1056];u=pre.encode(h*(1-m)+noise);target=pre.encode(h)
   assert np.array_equal(a["input"][i].numpy(),u) and np.array_equal(a["target"][i].numpy(),target) and np.array_equal(a["mask"][i].numpy(),m)
   scale=np.sqrt(np.mean(u.astype(float)**2,axis=-1)).clip(1e-8);qt=torch.stack([q(torch.from_numpy(row/sc)) for row,sc in zip(u,scale)]).numpy()
   assert np.array_equal(a["q"][i].numpy(),qt)
   with torch.no_grad():bands=model.input_bands(torch.from_numpy(u[None]).cuda()).cpu().numpy()[0]
   # GPU FFT transforms can differ slightly with batch shape; record tolerance.
   diff=float(np.max(abs(bands-a["bands"][i].numpy())));assert diff<2e-5
   checked.append(dict(index=i,seed=seed,input_target_mask_q_bitwise_reproduced=True,bands_max_error=diff))
 hashes={}
 for file in sorted((OUT/"cache").glob("*.npy")):hashes[str(file.relative_to(OUT))]=sha(file)
 write(OUT/"cache_manifest.json",hashes);write(OUT/"cache_gates.json",dict(passed=True,checked_train_rows=checked,all_cache_files_hashed=True,test_read=False))
 print(checked,flush=True)
if __name__=="__main__":main()
