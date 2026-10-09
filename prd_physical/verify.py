"""Pre-generation physics gates; failures prevent full generation/training."""
import json,time
import numpy as np,lal,lalsimulation as ls
from pycbc.conversions import get_lm_f0tau
from scipy.signal import welch,csd
from .physics import Simulator,one_sided_noise,amplitudes,FS,N,CONFIG

def verify(out):
 out.mkdir(parents=True,exist_ok=True);sim=Simulator();report={};t=time.time()
 a,r=amplitudes(1.);assert r[0]==r[1]==0;report['equal_mass_odd_modes_zero']=True
 rng=np.random.default_rng(52);size=65536;target=1e-40;ratios=[];variance=[]
 for k in range(16):
  x=one_sided_noise(np.full((2,size//2+1),target),FS,size,rng)
  f,p=welch(x,FS,nperseg=4096,axis=-1);ratios.extend((p[:,100:-100].mean(1)/target).tolist());variance.extend((x.var(1)/(target*FS/2)).tolist())
 report['white_psd_ratio']=float(np.mean(ratios));report['white_variance_ratio']=float(np.mean(variance));assert abs(report['white_psd_ratio']-1)<.05 and abs(report['white_variance_ratio']-1)<.05
 qnm=[]
 for mass in [1.9e5,3.4e7]:
  for spin in [.54,.69]:
   for l,m in [(2,2),(2,1),(3,3),(4,4)]:
    f,tau=get_lm_f0tau(mass,spin,l,m);vec=lal.CreateCOMPLEX16Vector(1);ls.SimIMREOBGenerateQNMFreqV2fromFinal(vec,mass,spin,l,m,1)
    errors=[abs(f/(vec.data[0].real/(2*np.pi))-1),abs(tau/(1/abs(vec.data[0].imag))-1)];assert max(errors)<.002;qnm.append(max(errors))
 report['qnm_max_relative_error']=max(qnm)
 pars=sim.parameters(42);h,p=sim.signal(pars);large=np.array([5e6,.3,6.,.8,1.,.4,.3,.1]);_,corner=sim.signal(large)
 report['largest_corner_tau_seconds']=float(corner[12]);report['worst_tail_damping_times']=(N/FS-2048-600)/corner[12];assert report['worst_tail_damping_times']>6
 half,_=sim.signal(pars,distance_multiplier=2.)
 report['distance_scaling_relative_l2']=float(np.linalg.norm(half-h/2)/np.linalg.norm(h/2));assert report['distance_scaling_relative_l2']<1e-10
 # Parameter-derived remnant redshift scaling is verified independently of strain orientation.
 mf=p[8];assert np.isclose(p[9],mf*(1+p[2]))
 bigger=Simulator(pad_seconds=4096.);h2,_=bigger.signal(pars);padding=float(np.linalg.norm(h2-h)/np.linalg.norm(h));report['padding_relative_l2']=padding;assert padding<1e-4
 higher=Simulator(order=35);h3,_=higher.signal(pars);interp=float(np.linalg.norm(h3-h)/np.linalg.norm(h));report['interpolation_relative_l2']=interp;assert interp<.01
 report['padding_accepted_seconds']=2048.;report['interpolation_order']=25
 # Colored-noise validation on the LONG FFT grid avoids conflating Welch
 # spectral convolution around response nulls with a pointwise PSD estimate.
 power=np.zeros_like(sim.psd);energy=[];cross=[]
 for k in range(128):
  long=one_sided_noise(sim.psd,FS,sim.long_n,np.random.default_rng(600+k));ft=np.fft.rfft(long,axis=-1);pwr=2*abs(ft)**2/(FS*sim.long_n);power+=pwr;cross.append(ft[0]*ft[1].conj());energy.append(long.var(1))
 band=(np.fft.rfftfreq(sim.long_n,1/FS)>.001)&(np.fft.rfftfreq(sim.long_n,1/FS)<.8)&(sim.psd.min(0)>1e-44)
 ratio=(power/128/sim.psd.clip(1e-100))[:,band].mean(1);var=np.mean(energy,axis=0)/(sim.psd[:,1:-1].sum(1)*FS/sim.long_n)
 report['colored_psd_ratios']=ratio.tolist();report['colored_variance_ratios']=var.tolist();assert np.max(abs(ratio-1))<.05 and np.max(abs(var-1))<.05
 coherence=np.abs(np.mean(cross,axis=0))**2/(power[0]/128*FS*sim.long_n/2*power[1]/128*FS*sim.long_n/2+1e-200)
 report['AE_mean_cross_coherence']=float(coherence[band].mean());assert report['AE_mean_cross_coherence']<.025
 report.update(passed=True,seconds=time.time()-t,config=CONFIG)
 (out/'physics_gates.json').write_text(json.dumps(report,indent=2));return report
