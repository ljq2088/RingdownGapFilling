"""Scale-equivariance diagnostic at the frozen fixed66 predictor."""
import numpy as np,torch
from prd_lowrank.settings import ROOT,CACHE,PROBE,RANK
from prd_lowrank.common import load_backbone,inferred
from prd_lowrank.data import packet,reconstruct
from prd_strategy.latent import CoefficientAdapter
from prd_physical.common import write
OUT=ROOT/'FillingGap/results_prd_bottleneck_20261013'
def main():
    torch.set_num_threads(2);model=load_backbone();head=CoefficientAdapter(RANK).eval();head.load_state_dict(torch.load(ROOT/'FillingGap/results_prd_lowrank_20261013/noise_augmented/best.pth',map_location='cpu',weights_only=True)['state_dict'])
    basis=np.load(PROBE/'training_direction_basis.npz')['vectors'][:,:RANK];B=torch.tensor(basis,dtype=torch.float32);u=np.load(CACHE/'train_input.npy',mmap_mode='r')[:32];m=np.load(CACHE/'train_mask.npy',mmap_mode='r')[:32]
    def predict(x):
        p=inferred(model,x,m);v={k:torch.from_numpy(a) for k,a in packet(x,p,m,basis,np.ones(len(x))).items()}
        with torch.no_grad():return reconstruct(head,v,B).numpy()
    reference=predict(u).astype(float);rows=[]
    for a in (.5,2.):
        p=predict((u*np.float32(a)).astype('f4')).astype(float);error=np.linalg.norm(p-a*reference,axis=(1,2))/np.linalg.norm(a*reference,axis=(1,2));assert error.max()<1e-5
        rows.append(dict(input_scale=a,max_relative_l2_error_to_scaled_prediction=float(error.max()),mean_error=float(error.mean())))
    write(OUT/'scale_homogeneity.json',dict(passed=True,scope='32 training observations with fixed66; CWT and bands regenerated after positive input scaling; frozen predictor; numerical output equivariance',rows=rows,interpretation='the normalized features are almost scale invariant, while prediction amplitude is restored via observed scale; inference of noise level currently relies on normalized waveform structure; explicitly supplying absolute whitened RMS and expected noise scale is a separate candidate requiring validation',current_training_unchanged=True,test_read=False))
    print(rows)
if __name__=='__main__':main()
