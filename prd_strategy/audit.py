"""Independent metric/prototype and population audits on all validation rows."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'2')
import json
import numpy as np,torch
from prd_physical.common import Preprocess,metric,summary,write,N,FS
from .settings import *
from .latent import WholeWindowAdapter,coordinates,RANK
from .analyze import full_summary,remove_endpoints

def independent_overlap(h,p,psd):
 # Explicit channel/frequency summation with the one-sided inner-product factor.
 a=np.fft.rfft(h);b=np.fft.rfft(p);factor=4/(FS*N)
 aa=np.zeros(len(h));bb=aa.copy();ab=aa.copy()
 for channel in range(2):
  weights=factor/psd[channel,1:-1]
  aa+=(np.abs(a[:,channel,1:-1])**2*weights).sum(-1)
  bb+=(np.abs(b[:,channel,1:-1])**2*weights).sum(-1)
  ab+=(np.real(a[:,channel,1:-1]*b[:,channel,1:-1].conj())*weights).sum(-1)
 return 1-np.clip(ab/np.sqrt(aa*bb),-1,1)

def main():
 torch.set_num_threads(2);cal=np.load(CAL);pre=Preprocess(cal['psd'],'white');truth=np.load(CACHE/'validation_truth.npy').astype(float)
 hf=np.fft.rfft(truth);snr=np.sqrt((4/(FS*N)*(abs(hf[...,1:-1])**2/cal['psd'][:,1:-1])).sum((1,2)))
 h=remove_endpoints(np.load(CACHE/'validation_target.npy')).reshape(VAL_N,-1)
 p=np.load(OUT/'frequency_snapshot_validation_prediction.npy');b=np.load(OUT/'training_direction_basis.npz')['vectors'][:,:RANK]
 orth=float(np.max(abs(b.T@b-np.eye(RANK))));endpoint=float(np.max(abs(np.fft.rfft(b.T.reshape(RANK,2,N))[:,:,[0,-1]])));assert orth<1e-12 and endpoint<1e-10
 projected=h@b@b.T;loss=((h-projected)**2).sum(1)/(h*h).sum(1)
 params=np.load(CACHE/'validation_parameters.npy');reports={};base=metric(truth,pre.decode(p),cal['psd'])
 kinds=['projection','legacy','balanced','balanced_augmented','balanced_repeated','direct_spectral','balanced_augmented_rank256']
 kinds += sorted(d.name.removeprefix('head_') for d in OUT.glob('head_*_seed*'))
 for kind in kinds:
  if kind=='projection':
   arr=np.load(OUT/'validation_prototype_predictions.npz')['frequency_snapshot__direction_basis']
  else:
   path=OUT/('head_'+kind)/'validation_prediction.npy'
   if not path.exists():continue
   arr=np.load(path)
  physical=pre.decode(arr);mm=metric(truth,physical,cal['psd']);check=independent_overlap(truth,physical,cal['psd']);diff=float(abs(mm-check).max());assert diff<1e-12
  rep=full_summary(truth,arr,pre,cal,snr);rep.update(independent_metric_max_absolute_difference=diff,improved_sources=int((mm<base).sum()),worsened_sources=int((mm>base).sum()),reduction_percent=100*(base.mean()-mm.mean())/base.mean(),strata={})
  for label,ix in [('snr_lt20',snr<20),('snr20_50',(snr>=20)&(snr<50)),('snr_ge50',snr>=50),('source_mass_lt5e5',params[:,0]<5e5),('source_mass_5e5_2e6',(params[:,0]>=5e5)&(params[:,0]<2e6)),('source_mass_ge2e6',params[:,0]>=2e6)]:
   if ix.any():rep['strata'][label]=dict(n=int(ix.sum()),baseline_mean=float(base[ix].mean()),candidate_mean=float(mm[ix].mean()))
  reports[kind]=rep
 # Torch integration reproduces the CPU pilot and exposes gradients to the DGF.
 obs=np.load(CACHE/'validation_input.npy')[:16];mask=np.load(CACHE/'validation_mask.npy')[:16];npcoord=coordinates(obs,p[:16],mask,b)
 module=WholeWindowAdapter(b);x=torch.tensor(obs);y=torch.tensor(p[:16],requires_grad=True);m=torch.tensor(mask)
 initial=module(x,y,m);expected=(npcoord['coefficients']@b.T).reshape(-1,2,N)+npcoord['endpoint']
 relative=float(np.linalg.norm(initial.detach().numpy()-expected)/np.linalg.norm(expected));assert relative<1e-6
 initial.square().mean().backward();headgrad=float(module.coefficient_adapter.head.weight.grad.norm());backgrad=float(y.grad.norm());assert headgrad>0 and backgrad>0
 state=torch.load(OUT/'head_balanced/best.pth',map_location='cpu',weights_only=True);module.coefficient_adapter.load_state_dict(state['state_dict']);got=module(x,y.detach(),m).detach().numpy();cpu=np.load(OUT/'head_balanced/validation_prediction.npy')[:16];trained_relative=float(np.linalg.norm(got-cpu)/np.linalg.norm(cpu));assert trained_relative<1e-6
 write(OUT/'prototype_audit.json',dict(passed=True,all_validation_rows_retained=True,n=VAL_N,test_read=False,basis_orthogonality_max_error=orth,basis_endpoint_max_error=endpoint,validation_target_discarded_direction_energy=summary(loss),basis_rank=RANK,torch_module=dict(parameters=sum(a.numel() for a in module.parameters()),initial_projection_relative_error=relative,pilot_reproduction_relative_error=trained_relative,head_gradient_norm=headgrad,backbone_output_gradient_norm=backgrad),candidates=reports,selection_scope='descriptive validation diagnostics after validation-based model selection; independent test required'))
 print(json.dumps({k:dict(psd=v['psd']['mean'],time=v['time']['mean'],hann=v['hann']['mean'],gain=v['reduction_percent'],improved=v['improved_sources']) for k,v in reports.items()},indent=2))
if __name__=='__main__':main()
