from pathlib import Path
import json,sys,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/home/ljq/code/RingdownGapFilling_PRD');sys.path.insert(0,str(ROOT))
from prd_physical.physics import Simulator,amplitudes,FS,N,REF_SECONDS
from prd_physical.common import metric
from prd_physical.prepare import crop_meta
from prd_conditioned.settings import OUT,CAL
from pycbc.waveform.ringdown import get_td_from_final_mass_spin,td_damped_sinusoid,Kerr_factor
from pycbc.conversions import get_lm_f0tau
from lisatools.utils.utility import AET
HERE=Path(__file__).resolve().parent;dest=HERE/'waveform_trace';dest.mkdir(exist_ok=True)
pars=np.load(OUT/'fresh_test/parameters.npy');seeds=np.load(OUT/'fresh_test/seeds.npy');truth=np.load(OUT/'fresh_test/truth.npy',mmap_mode='r');psd=np.load(CAL)['psd']
sim=Simulator();result=[];fig,axes=plt.subplots(3,2,figsize=(12,9),layout='constrained');colors=['#277c91','#203a59','#c46c26','#648833'];names=['Exponential k=5','Current k=10','Exponential k=20','C1 envelope join']
for row,index in enumerate([1116,1885,1777]):
    p=pars[index];total,q,z,iota,lam,beta,psi,phase=p[:8];a,rel=amplitudes(q)
    hp,hc=get_td_from_final_mass_spin(final_mass=p[9],final_spin=p[10],distance=p[11],lmns=['221','211','331','441'],amp220=a,amp210=rel[0],amp330=rel[1],amp440=rel[2],phi220=2*phase,phi210=phase,phi330=3*phase,phi440=4*phase,inclination=iota,delta_t=1/FS,t_final=(sim.long_n/FS-REF_SECONDS-sim.pad/FS),harmonics='spherical',taper=True)
    times=hp.sample_times.numpy();negative=times<0;norm=Kerr_factor(p[9],p[11]);source=[];outputs=[];crop,m=crop_meta(p,int(seeds[index]));post_error=[]
    for variant in (5,10,20,'C1'):
        complex_source=np.zeros(len(hp),complex)
        for (l,k),alpha in zip([(2,2),(2,1),(3,3),(4,4)],a*np.r_[1,rel]):
            f,tau=get_lm_f0tau(p[9],p[10],l,k)
            plus,cross=td_damped_sinusoid(f,tau,alpha,k*phase,times.copy(),l=l,m=k,inclination=iota,harmonics='spherical')
            scale=np.ones(len(times));x=times[negative]/tau
            scale[negative]=np.exp((variant-10)*x) if variant!='C1' else np.exp(-11*x-50*x*x)
            complex_source+=(plus+1j*cross)*scale
        complex_source*=norm
        error=float(np.linalg.norm(complex_source[~negative]-(np.asarray(hp)+1j*np.asarray(hc))[~negative])/np.linalg.norm(np.asarray(hp)[~negative]+1j*np.asarray(hc)[~negative]));assert error<2e-12,error;post_error.append(error)
        strain=np.zeros(sim.long_n,complex);start=sim.pad+int(REF_SECONDS*FS)+int(round(float(hp.start_time)*FS));count=min(len(hp),sim.long_n-start);strain[start:start+count]=complex_source[:count]*np.exp(-2j*psi)
        sim.response.get_projections(strain,lam,beta,t0=sim.pad/FS);ae=np.asarray(AET(*sim.response.get_tdi_delays())[:2])[:,sim.pad:sim.pad+N].copy();outputs.append(ae[:,crop:crop+1056]);source.append(complex_source)
    old=np.asarray(truth[index],float);reference=outputs[1];reproduction=float(np.linalg.norm(reference-old)/np.linalg.norm(old));assert reproduction<1e-6,reproduction
    _,tau=get_lm_f0tau(p[9],p[10],2,2);axis=axes[row,0];source_scale=max(abs(np.asarray(hp)+1j*np.asarray(hc)))
    for j,x in enumerate(source):axis.plot(times,x.real/source_scale,color=colors[j],lw=1.1,label=names[j])
    axis.set(xlim=(-.5*tau,.7*tau),title=f'Seed {seeds[index]} | source h+',ylabel='Normalized amplitude');axis.axvline(0,color='#718096',ls=':',lw=.8);axis.grid(alpha=.15)
    axis=axes[row,1];scale=abs(old).max();channel=int(np.argmax((old*old).sum(-1)))
    for j,x in enumerate(outputs):axis.plot(np.arange(1056)/FS,x[channel]/scale,color=colors[j],lw=1.1,label=names[j])
    axis.set(xlim=(0,250),title=f'TDI {"AE"[channel]} | fixed original crop');axis.grid(alpha=.15)
    if row==0:
        axes[row,0].legend(fontsize=8,frameon=False);axes[row,1].legend(fontsize=8,frameon=False)
    if row==2:axes[row,0].set_xlabel('Source time relative to reference (s)');axes[row,1].set_xlabel('Detector analysis-window time (s)')
    result.append(dict(seed=int(seeds[index]),post_reference_source_relative_error=post_error,tdi_original_reproduction_relative_error=reproduction,direct_psd_mismatch_vs_current={name:float(metric(old[None],x[None],psd)[0]) for name,x in zip(names,outputs)},physical_relative_l2_vs_current={name:float(np.linalg.norm(x-old)/np.linalg.norm(old)) for name,x in zip(names,outputs)},interpretation='All variants preserve the post-reference QNM signal. Only the numerical pre-reference extension changes; this is a sensitivity test, not a proposed astrophysical pre-merger model.'))
fig.savefig(dest/'onset_sensitivity.png',dpi=160,facecolor='white');plt.close(fig);(dest/'onset_sensitivity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
