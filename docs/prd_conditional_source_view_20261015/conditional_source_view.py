import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import sys,json,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;R=Path('/home/ljq/code/RingdownGapFilling_PRD');sys.path.insert(0,str(R))
import prd_physical.physics as physics
from prd_physical.prepare import crop_meta
from pycbc.waveform.ringdown import get_td_from_final_mass_spin
from pycbc.conversions import get_lm_f0tau
data=np.load(P.parent/'ringdown_frequency_examples_20261015/figure_native_samples.npz')
psd=np.load(R/'FillingGap/results_prd_physical_transfer_20261009/calibration_psd.npz')['psd']
weights=1/np.sqrt(psd);weights[:,0]=0;weights[:,-1]=0
sim=physics.Simulator();mode=None;quadrature=0;capture={}
def generator(**kw):
 hp,hc=get_td_from_final_mass_spin(**kw)
 if mode is not None:
  k='phi'+mode;kp=kw.copy();km=kw.copy();kp[k]+=quadrature*np.pi/2;km[k]+=quadrature*np.pi/2+np.pi
  ap,ac=get_td_from_final_mass_spin(**kp);bp,bc=get_td_from_final_mass_spin(**km)
  hp=(ap-bp)*.5;hc=(ac-bc)*.5
 capture.update(time=np.array(hp.sample_times),hp=np.array(hp),hc=np.array(hc))
 return hp,hc
physics.get_td_from_final_mass_spin=generator
def whiten(v):
 z=np.fft.rfft(v,axis=-1)*weights
 return np.concatenate([z.real.ravel(),z.imag.ravel()])
def solve(B,y):
 scales=np.linalg.norm(B,axis=0);X=B/scales
 coef,_,rank,sv=np.linalg.lstsq(X,y,rcond=1e-10)
 return coef/scales,int(rank),float(sv[0]/sv[-1])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'path.simplify':False})
fig,axes=plt.subplots(2,3,figsize=(10.4,5.0),layout='constrained');reports=[];saved={}
for j,(idx,seed) in enumerate(zip(data['index'],data['seeds'])):
 mode=None;pars=sim.parameters(int(seed));full,meta=sim.signal(pars);origin,_=crop_meta(meta,int(seed));truth=full[:,origin:origin+1056]
 assert np.array_equal(truth.astype(np.float32).astype(float),data['truth'][j])
 source=np.stack([capture['hp'],capture['hc']]);ts=capture['time'].copy();tdi_basis=[];src_basis=[]
 for mode in ['220','210','330','440']:
  for quadrature in [0,1]:
   response,_=sim.signal(pars);tdi_basis.append(response[:,origin:origin+1056]);src_basis.append(np.stack([capture['hp'],capture['hc']]))
 mode=None;tdi_basis=np.array(tdi_basis);src_basis=np.array(src_basis)
 coeff_exact=np.tile([1.,0.],4)
 linearity=float(np.linalg.norm(np.einsum('k,kct->ct',coeff_exact,tdi_basis)-truth)/np.linalg.norm(truth))
 B=np.stack([whiten(v) for v in tdi_basis],axis=1)
 coef,rank,condition=solve(B,whiten(data['reconstruction'][j]))
 ctr,_,_=solve(B,whiten(truth));target_recovery=np.einsum('k,kct->ct',ctr,src_basis)
 source_rec=np.einsum('k,kct->ct',coef,src_basis);response_rec=np.einsum('k,kct->ct',coef,tdi_basis)
 # Control: fit the same known response family directly to the original masked observation.
 G=1-data['mask'][j];Bg=np.stack([whiten(v*G) for v in tdi_basis],axis=1)
 cb,rank_b,condition_b=solve(Bg,whiten(data['observation'][j]));source_baseline=np.einsum('k,kct->ct',cb,src_basis)
 f,tau=get_lm_f0tau(float(meta[9]),float(meta[10]),2,2);use=(ts>=0)&(ts<=4*tau)
 err=float(np.linalg.norm(source_rec[:,use]-source[:,use])/np.linalg.norm(source[:,use]));base_err=float(np.linalg.norm(source_baseline[:,use]-source[:,use])/np.linalg.norm(source[:,use]));roundtrip=float(np.linalg.norm(target_recovery[:,use]-source[:,use])/np.linalg.norm(source[:,use]))
 projection_error=float(np.linalg.norm(whiten(response_rec-data['reconstruction'][j]))/np.linalg.norm(whiten(data['reconstruction'][j])))
 report=dict(index=int(idx),seed=int(seed),frequency_hz=float(f),tau_seconds=float(tau),rank=rank,column_normalized_condition=condition,clean_response_linearity_error=linearity,clean_source_roundtrip_relative_error=roundtrip,conditional_source_joint_relative_error=err,conditional_direct_observation_fit_error=base_err,observation_fit_rank=rank_b,observation_fit_condition=condition_b,dgf_response_projection_relative_error=projection_error,coefficient_fit=coef.tolist())
 reports.append(report);print(json.dumps(report),flush=True)
 exponent=int(np.floor(np.log10(abs(source[:,use]).max())));scale=10.**exponent
 for c in range(2):
  ax=axes[c,j];ax.plot(ts[use]/tau,source[c,use]/scale,color='#203d59',lw=1.6,label='Injected strain');ax.plot(ts[use]/tau,source_rec[c,use]/scale,color='#d16b2e',ls=(0,(4,2)),lw=1.4,label='From DGF A/E (conditional fit)');ax.axhline(0,color='#b7bfc7',lw=.5,zorder=0);ax.set(xlim=(0,4),xlabel=r'$t/\tau_{220}$',ylabel=('$h_+$' if c==0 else r'$h_\times$')+rf' ($10^{{{exponent}}}$)',title=f'({chr(97+c*3+j)}) Source {j+1} | '+rf'$f_{{220}}={1000*f:.1f}$ mHz');ax.grid(alpha=.13)
 saved[f'source{j+1}_time']=ts[use];saved[f'source{j+1}_target']=source[:,use];saved[f'source{j+1}_conditional_reconstruction']=source_rec[:,use];saved[f'source{j+1}_conditional_observation_fit']=source_baseline[:,use];saved[f'source{j+1}_response_basis']=tdi_basis
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside upper center',ncols=2,frameon=False)
for ext in ['png','pdf','svg']:fig.savefig(P/('conditional_source_strain.'+ext),dpi=200)
np.savez_compressed(P/'conditional_source_arrays.npz',**saved)
out=dict(method='Noise-weighted least-squares projection of frozen DGF A/E output onto eight exact response basis functions, followed by source-strain synthesis. Four injected fundamental modes, two phase quadratures per mode.',conditions='Injected detector-frame mass, remnant spin, sky position, inclination, polarization, arrival epoch, and the simulated ringup history are fixed. Coefficients are fitted from the model output. This is a conditional source reconstruction with a strong waveform prior, not an independent source-parameter recovery.',source_display='Post-reference 0 to 4 tau220; h+ and hcross use the original 0.5-s source sampling. The reference epoch is the source model t=0. The A/E suppression interval has a distributed source-time response, so no single source-time gap is shaded.',weighted_fit='Both A/E channels jointly; frozen calibration PSD; DC and Nyquist excluded; column-normalized least squares rcond=1e-10.',sources=reports)
(P/'conditional_source_report.json').write_text(json.dumps(out,indent=2))
