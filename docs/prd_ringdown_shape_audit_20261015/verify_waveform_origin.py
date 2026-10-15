import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import sys,json,numpy as np
from scipy.signal import hilbert
P=Path(__file__).resolve().parent;R=Path('/home/ljq/code/RingdownGapFilling_PRD');sys.path.insert(0,str(R))
import prd_physical.physics as physics
from prd_physical.prepare import crop_meta
from pycbc.conversions import get_lm_f0tau
from pycbc.waveform.ringdown import get_td_from_final_mass_spin
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
a=np.load(P/'figure_native_samples.npz');T=R/'FillingGap/results_prd_tail_20261015/fresh_test';pars=np.load(T/'parameters.npy',mmap_mode='r')
captured={}
def capture(**kw):
    hp,hc=get_td_from_final_mass_spin(**kw)
    captured.update(hp=np.asarray(hp).copy(),hc=np.asarray(hc).copy(),time=np.asarray(hp.sample_times).copy())
    return hp,hc
physics.get_td_from_final_mass_spin=capture
sim=physics.Simulator();reports=[];arrays={}
fig,axes=plt.subplots(2,3,figsize=(12,5.7),layout='constrained')
def basis(t,ft):
    return np.array([np.exp(-t/tau)*trig(2*np.pi*f*t) for f,tau in ft for trig in [np.cos,np.sin]]).T
for j,(i,seed) in enumerate(zip(a['index'],a['seeds'])):
    full,meta=sim.signal(sim.parameters(int(seed)));origin,mask=crop_meta(meta,int(seed));crop=full[:,origin:origin+1056]
    assert np.array_equal(crop.astype(np.float32).astype(float),a['truth'][j])
    assert np.array_equal(meta,pars[i])
    nv=sim.noise(int(seed)+100000000)[:,origin:origin+1056]
    assert np.array_equal(nv.astype(np.float32).astype(float),a['noise'][j])
    ft=[get_lm_f0tau(float(meta[9]),float(meta[10]),l,m) for l,m in [(2,2),(2,1),(3,3),(4,4)]]
    f,tau=ft[0];tt=a['time'];peak=float(tt[np.sum(abs(hilbert(a['truth'][j]))**2,axis=0).argmax()])
    gate=(tt>=peak+80)&(tt<=min(527.5,peak+300));x=tt[gate]-(peak+80);B=basis(x,ft)
    fits=[]
    for c in range(2):
        v=crop[c,gate];scale=max(abs(v));coef=np.linalg.lstsq(B,v/scale,rcond=None)[0];fit=B@coef*scale
        fits.append(float(np.linalg.norm(v-fit)/np.linalg.norm(v)))
        ax=axes[j,c+1];ax.plot(tt-peak,crop[c]/abs(crop[c]).max(),color='#183b60',label='TDI target');ax.plot(tt-peak,a['reconstruction'][j,c]/abs(crop[c]).max(),color='#c46023',ls='--',label='DGF');ax.plot(tt[gate]-peak,fit/abs(crop[c]).max(),color='#1c897e',ls=':',label='Known-frequency QNM fit');ax.set(xlim=(-80,300),xlabel=r'$t-t_{\rm pk}$ (s)',ylabel='Normalized fractional frequency',title=f'Source {j+1} | {"AE"[c]}');ax.grid(alpha=.15)
    ts=captured['time'];use=(ts>=0)&(ts<=4*tau);hp=captured['hp'][use];hpscale=abs(hp).max();B=basis(ts[use],ft);fit=B@np.linalg.lstsq(B,hp/hpscale,rcond=None)[0]*hpscale
    source_error=float(np.linalg.norm(hp-fit)/np.linalg.norm(hp));ax=axes[j,0];ax.plot(ts[use]/tau,hp/hpscale,color='#183b60');ax.set(xlabel=r'$t/\tau_{220}$',ylabel=r'$h_+/\max|h_+|$',title=f'Source {j+1} | input polarization');ax.grid(alpha=.15)
    reports.append(dict(index=int(i),seed=int(seed),truth_reproduced_bitwise=True,noise_reproduced_bitwise=True,detector_remnant_mass_msun=float(meta[9]),spin=float(meta[10]),f220_hz=float(f),tau220_s=float(tau),cycles_per_efold=float(f*tau),samples_per_cycle=float(2/f),source_fit_relative_error=source_error,tdi_late_fit_relative_error=fits,tdi_fit_interval_s=[float(tt[gate][0]),float(tt[gate][-1])],joint_peak_s=peak,tdi_max_abs=float(abs(crop).max())))
    arrays[f'source{j+1}_time_s']=ts[use];arrays[f'source{j+1}_hp']=hp
axes[0,2].legend(fontsize=7);fig.savefig(P/'source_tdi_diagnostic.png',dpi=180)
np.savez_compressed(P/'source_polarizations.npz',**arrays)
out=dict(reference='https://arxiv.org/abs/1305.4306, Fig.1: Schwarzschild Zerilli multipoles, radial infall E=1, 2M=1, cumulative overtones n=0..3',source_taper='exp(10t/tau) before t=0; exp(-t/tau) after t=0; one-sided envelope derivatives differ by 11 A/tau',fits='Eight real sine/cosine basis functions at four prescribed fundamental QNM frequencies and damping times. Truth fits diagnose simulated waveform content; they do not replace model predictions.',sources=reports)
(P/'waveform_origin_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
