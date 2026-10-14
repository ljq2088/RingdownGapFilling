from pathlib import Path
import json,sys,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/home/ljq/code/RingdownGapFilling_PRD');sys.path.insert(0,str(ROOT))
from prd_physical.physics import Simulator,amplitudes,FS,N,REF_SECONDS,COSMO
from prd_physical.prepare import crop_meta
from prd_conditioned.settings import OUT
from pycbc.waveform.ringdown import get_td_from_final_mass_spin
from lisatools.utils.utility import AET
from pycbc.conversions import get_lm_f0tau
HERE=Path(__file__).resolve().parent;indices=list(map(int,sys.argv[1:])) if len(sys.argv)>1 else [1116,1885,1777];dest=HERE/('waveform_trace_oscillatory' if len(sys.argv)>1 else 'waveform_trace');dest.mkdir(exist_ok=True)
pars=np.load(OUT/'fresh_test/parameters.npy');seeds=np.load(OUT/'fresh_test/seeds.npy');truth=np.load(OUT/'fresh_test/truth.npy',mmap_mode='r')
sim=Simulator();reports=[];fig,axes=plt.subplots(3,2,figsize=(13,9.5),layout='constrained')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
for row,index in enumerate(indices):
    p=pars[index];total,q,z,iota,lam,beta,psi,phase=p[:8];a,rel=amplitudes(q)
    hp,hc=get_td_from_final_mass_spin(final_mass=p[9],final_spin=p[10],distance=p[11],lmns=['221','211','331','441'],amp220=a,amp210=rel[0],amp330=rel[1],amp440=rel[2],phi220=2*phase,phi210=phase,phi330=3*phase,phi440=4*phase,inclination=iota,delta_t=1/FS,t_final=(sim.long_n/FS-REF_SECONDS-sim.pad/FS),harmonics='spherical',taper=True)
    strain=np.zeros(sim.long_n,complex);start=sim.pad+int(REF_SECONDS*FS)+int(round(float(hp.start_time)*FS));count=min(len(hp),sim.long_n-start)
    strain[start:start+count]=(np.asarray(hp)[:count]+1j*np.asarray(hc)[:count])*np.exp(-2j*psi)
    sim.response.get_projections(strain,lam,beta,t0=sim.pad/FS)
    ae=np.asarray(AET(*sim.response.get_tdi_delays())[:2])[:,sim.pad:sim.pad+N].copy()
    crop,m=crop_meta(p,int(seeds[index]));err=float(abs(ae[:,crop:crop+1056].astype('f4')-truth[index]).max());assert err==0,err
    # A step probes the support of the onset response, without altering data.
    onset=[]
    for polarization in (1.,1j):
        step=np.zeros(sim.long_n,complex);step[sim.pad+int(REF_SECONDS*FS):]=polarization
        sim.response.get_projections(step,lam,beta,t0=sim.pad/FS)
        onset.append(np.asarray(AET(*sim.response.get_tdi_delays())[:2])[:,sim.pad:sim.pad+N].copy())
    onset=np.stack(onset)
    amplitude=np.sqrt((onset**2).sum((0,1)));support=amplitude>amplitude.max()*1e-6;positions=np.flatnonzero(support)
    bounds=[int(positions[0]),int(positions[-1])];trans=np.zeros(N,bool);trans[bounds[0]:bounds[1]+1]=True
    energy=(ae**2).sum(0);full=energy.sum();captured=energy[crop:crop+1056].sum()/full
    f,tau=get_lm_f0tau(p[9],p[10],2,2);time=np.arange(N)/FS-REF_SECONDS;source=strain[sim.pad:sim.pad+N]
    xmin=-150.;xmax=max(800.,6*tau+100)
    for col,data in enumerate([np.stack((source.real,source.imag)),ae]):
        ax=axes[row,col];scale=abs(data).max();shift=0 if col==0 else time[bounds[0]];display_time=time-shift
        if col==1:ax.axvspan(crop/FS-REF_SECONDS-shift,(crop+1056)/FS-REF_SECONDS-shift,color='#dfe7f0',label='Current 528-s window')
        ax.plot(display_time,data[0]/scale,color='#203a59',lw=1,label=r'$h_+$' if col==0 else 'A')
        ax.plot(display_time,data[1]/scale,color='#bd5d22',lw=.9,alpha=.9,label=r'$h_\times$' if col==0 else 'E')
        ax.axvline(0,color='#4b677e',lw=.7,ls=':')
        if col==1:ax.axvspan(0,time[bounds[1]]-shift,color='#d48b3b',alpha=.16,label='Response to onset step')
        ax.set(xlim=(xmin,xmax),ylim=(-1.12,1.12),ylabel='Normalized amplitude',title=f'Seed {seeds[index]} | '+('Source polarizations' if col==0 else 'TDI A/E response'))
        ax.grid(alpha=.15);ax.legend(loc='upper right',fontsize=8,frameon=False)
        if row==2:ax.set_xlabel('Time from source reference (s)' if col==0 else 'Time from onset-response start at detector (s)')
    reports.append(dict(index=index,seed=int(seeds[index]),source_period_220_seconds=float(1/f),tau_220_seconds=float(tau),stored_crop_bitwise_reproduced=True,source_epoch_alignment_max_error=err,source_reference_ssb_seconds=REF_SECONDS,analysis_window_ssb_seconds=[crop/FS-REF_SECONDS,(crop+1056)/FS-REF_SECONDS],onset_response_support_ssb_seconds=[time[bounds[0]],time[bounds[1]]],fraction_parent_time_energy_captured_by_crop=float(captured),fraction_parent_time_energy_in_onset_step_support=float(energy[trans].sum()/full),note='Onset-step support is a numerical response diagnostic, not a decomposition into separate physical modes. SSB and constellation arrival times differ by the plane-wave light-travel time.'))
    np.savez_compressed(dest/f'trace_{seeds[index]}.npz',time_ssb=time,source=source,tdi=ae,onset_step_response=onset,crop_start=crop,parameters=p)
fig.savefig(dest/'source_to_tdi.png',dpi=160,facecolor='white');plt.close(fig)
(dest/'source_to_tdi.json').write_text(json.dumps(reports,indent=2));print(json.dumps(reports,indent=2))
