"""Physical strain -> finite-arm LISA response -> orthonormal TDI2 A/E.

All masses entering QNMs are detector-frame REMNANT solar masses. The input
mass prior is SOURCE-frame progenitor total mass. No ad-hoc strain rescaling.
The idealized equal-arm stationary instrument noise excludes the foreground.
"""
import numpy as np
import lal,lalsimulation as ls
from astropy.cosmology import FlatLambdaCDM
from pycbc.waveform.ringdown import get_td_from_final_mass_spin
from pycbc.conversions import get_lm_f0tau
from fastlisaresponse import pyResponseTDI
from lisatools.detector import EqualArmlengthOrbits
from lisatools.sensitivity import A2TDISens,E2TDISens
from lisatools.utils.utility import AET

FS=2.;N=32768;PAD_SECONDS=2048.;REF_SECONDS=2048.;MISSION_EPOCH=31558149.7632
COSMO=FlatLambdaCDM(H0=67.4,Om0=.315,Tcmb0=0.)
PARAM_NAMES=['source_total_mass_msun','q_light_over_heavy','redshift','inclination','longitude','latitude','polarization','orbital_phase','source_remnant_mass_msun','detector_remnant_mass_msun','remnant_spin','distance_mpc','longest_tau_seconds','peak_index','reference_seconds']
CONFIG=dict(schema_version=1,source_total_mass_msun=[1e5,5e6],mass_prior='log-uniform',q_light_over_heavy=[.3,1.],redshift=[1.,6.],orientations='isotropic',progenitor_spins=[0.,0.],cosmology=dict(H0=67.4,Om0=.315),fs_hz=FS,n_samples=N,tdi='2nd generation',channels=['A','E'],aet_convention=['(Z-X)/sqrt(2)','(X-2Y+Z)/sqrt(6)','(X+Y+Z)/sqrt(3)'],noise_model='scirdv1',units='fractional frequency',observation='G*h+n',pad_seconds=PAD_SECONDS,reference_seconds=REF_SECONDS,amplitude_fit='Kamaretsos 1107.0854 Eqs31-34 at 10 remnant M after 22 luminosity peak',angular_harmonics='spherical',relative_mode_phases='m*orbital_phase; simplified coherent convention, not NR-fitted',mission_epoch_seconds=0.,foreground=False,gap_seconds=33.,natural_gap_offset_seconds=[-66.,66.],invalid_policy='abort; no outcome-based exclusion')

def amplitudes(q):
 if not .3<=q<=1.:raise ValueError('q outside validated nonspinning prior')
 qstar=1/q;a=.25*np.exp(-qstar/7.5)
 return a,[.13*np.sqrt(qstar-1),.18*(qstar-1)**(1/3),.024*qstar**.75]

def one_sided_noise(psd,fs,n,rng):
 """Interior Fourier quadratures each have variance N*fs*S/4."""
 psd=np.asarray(psd,dtype=float)
 assert psd.shape[-1]==n//2+1 and np.isfinite(psd).all() and (psd>=0).all()
 z=(rng.normal(size=psd.shape)+1j*rng.normal(size=psd.shape))*np.sqrt(n*fs*psd/4)
 z[...,0]=0
 if n%2==0:z[...,-1]=rng.normal(size=psd.shape[:-1])*np.sqrt(n*fs*psd[...,-1]/2)
 return np.fft.irfft(z,n=n,axis=-1)

class RebasedOrbits(EqualArmlengthOrbits):
 """Keep the physical observation epoch fixed when padding is changed."""
 def __init__(self,pad_seconds):
  self.epoch_shift=MISSION_EPOCH-pad_seconds;super().__init__()
 @property
 def t_base(self):return super().t_base-self.epoch_shift

CONFIG['mission_epoch_seconds']=MISSION_EPOCH
CONFIG['numerical_ringup']='PyCBC taper=True, short tau/10 pre-reference ringup; t>=0 amplitude unchanged; not an IMR model'

class Simulator:
 def __init__(self,pad_seconds=PAD_SECONDS,order=25):
  self.pad=int(round(pad_seconds*FS));self.long_n=N+2*self.pad
  orbit=RebasedOrbits(pad_seconds);orbit.configure(linear_interp_setup=True)
  self.response=pyResponseTDI(FS,self.long_n,order=order,tdi='2nd generation',tdi_chan='XYZ',orbits=orbit,force_backend='cpu')
  f=np.fft.rfftfreq(self.long_n,1/FS);self.psd=np.zeros((2,len(f)))
  for c,sens in enumerate([A2TDISens,E2TDISens]):self.psd[c,1:]=sens.get_Sn(f[1:],model='scirdv1')
  assert np.isfinite(self.psd).all() and (self.psd>=0).all()
 def noise(self,seed):
  noise=one_sided_noise(self.psd,FS,self.long_n,np.random.default_rng(seed))
  return noise[:,self.pad:self.pad+N].copy()
 def parameters(self,seed):
  rng=np.random.default_rng(seed)
  return np.array([10**rng.uniform(5,np.log10(5e6)),rng.uniform(.3,1),rng.uniform(1,6),np.arccos(rng.uniform(-1,1)),rng.uniform(0,2*np.pi),np.arcsin(rng.uniform(-1,1)),rng.uniform(0,np.pi),rng.uniform(0,2*np.pi)])
 def signal(self,pars,distance_multiplier=1.):
  total,q,z,iota,lam,beta,psi,phase=map(float,pars[:8]);m1=total/(1+q);m2=total-m1
  code,fraction,spin=ls.SimIMREOBFinalMassSpin(m1,m2,[0.,0.,0.],[0.,0.,0.],ls.SEOBNRv4)
  assert code==0 and 0<fraction<1 and 0<spin<1
  mf=total*fraction;mz=mf*(1+z);distance=float(COSMO.luminosity_distance(z).value)*distance_multiplier
  longest=max(get_lm_f0tau(mz,spin,l,m)[1] for l,m in [(2,2),(2,1),(3,3),(4,4)])
  assert (N/FS-REF_SECONDS-600)>6*longest,'Window inadequate for six damping times and constellation delay'
  a,relative=amplitudes(q)
  kw=dict(final_mass=mz,final_spin=spin,distance=distance,lmns=['221','211','331','441'],amp220=a,amp210=relative[0],amp330=relative[1],amp440=relative[2],phi220=2*phase,phi210=phase,phi330=3*phase,phi440=4*phase,inclination=iota,delta_t=1/FS,t_final=(self.long_n/FS-REF_SECONDS-self.pad/FS),harmonics='spherical',taper=True)
  hp,hc=get_td_from_final_mass_spin(**kw);strain=np.zeros(self.long_n,complex);start=self.pad+int(REF_SECONDS*FS)+int(round(float(hp.start_time)*FS));count=min(len(hp),self.long_n-start)
  strain[start:start+count]=(np.asarray(hp)[:count]+1j*np.asarray(hc)[:count])*np.exp(-2j*psi)
  self.response.get_projections(strain,lam,beta,t0=self.pad/FS)
  xyz=self.response.get_tdi_delays();h=np.asarray(AET(*xyz)[:2])[:,self.pad:self.pad+N].copy()
  assert np.isfinite(h).all() and np.linalg.norm(h)>0
  peak=int(np.argmax(np.sum(h*h,axis=0)))
  metadata=np.r_[pars,mf,mz,spin,distance,longest,peak,REF_SECONDS]
  return h,metadata
 def example(self,seed):
  h,p=self.signal(self.parameters(seed));n=self.noise(seed+100000000)
  return h.astype(np.float32),n.astype(np.float32),p

def gap_mask(peak,seed,targeted=False,length=66):
 rng=np.random.default_rng(seed)
 start=int(peak)-int(rng.integers(1,length)) if targeted else int(peak)+int(rng.integers(-132,133))
 start=int(np.clip(start,0,N-length));mask=np.zeros(N,np.float32);mask[start:start+length]=1
 return mask,start

def effective_psd(simulator,count=512,seed=2027000000):
 """PSD of the FINITE cropped process, including spectral leakage at TDI nulls.
 Independent pure-noise calibration; no signal or benchmark targets are read.
 It is frozen before training, shared by all models, and never fit to outcomes.
 """
 sums=np.zeros((2,N//2+1),float)
 for k in range(count):
  n=simulator.noise(seed+k);sums+=np.abs(np.fft.rfft(n,axis=-1))**2/(FS*N)
 psd=sums/count;psd[:,1:-1]*=2
 return psd

