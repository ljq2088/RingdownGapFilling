"""Read-only legacy audit and isolated LISA TDI reference smoke test.
Run from the WSL repository; never edits legacy datasets or checkpoints.
"""
from pathlib import Path
import sys,os,json,inspect,importlib.metadata
os.environ.setdefault('MPLBACKEND','Agg')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import numpy as np,h5py,lal,lalsimulation
from scipy.signal import welch
from gwpy.timeseries import TimeSeries
from pycbc import conversions
from pycbc.waveform.ringdown import get_td_from_final_mass_spin
from utils.noise import generate_noise_from_psd
from data import data_generation as legacy
from data import waveform as wf
from fastlisaresponse import pyResponseTDI
from lisatools.detector import EqualArmlengthOrbits
from lisatools.sensitivity import A2TDISens,E2TDISens,T2TDISens
from lisatools.utils.utility import AET
OUT=ROOT/'FillingGap/results_lisa_physics_audit_20261008';OUT.mkdir(exist_ok=True)
def noise_from_one_sided_psd(psd,fs,n,rng):
 z=(rng.normal(size=len(psd))+1j*rng.normal(size=len(psd)))*np.sqrt(n*fs*psd/4)
 z[0]=0
 if n%2==0:z[-1]=rng.normal()*np.sqrt(n*fs*psd[-1]/2)
 return np.fft.irfft(z,n=n)
def main():
 result={'versions':{k:importlib.metadata.version(k) for k in ['gwosc','gwpy','pycbc','lalsuite','fastlisaresponse']}}
 result['versions']['lisatools_distribution']=importlib.metadata.packages_distributions().get('lisatools',[])
 f=h5py.File(ROOT/'data/LISA_noise.mat');t=f['tVec'][:16].ravel();dt=float(np.diff(t).mean());result['legacy_noise_file']={'keys':list(f),'attrs':dict(f.attrs),'sample_interval_seconds':dt,'nyquist_hz':.5/dt,'channel':'A=(Z-X)/sqrt(2)','channel_units_and_tdi_generation':'not recorded in file metadata'}
 fi=np.arange(legacy.Config.f_in,legacy.Config.f_out,legacy.Config.f_step)
 result['legacy_psd']={'measured_max_hz':float(legacy.psd_interp_func.x[-1]),'fraction_generation_bins_outside_support':float((fi>legacy.psd_interp_func.x[-1]).mean()),'outside_support_psd':float(legacy.psd_interp_func.fill_value)}
 n=32768;fs=2.;S=1e-40;ratios=[];new=[]
 for seed in range(8):
  np.random.seed(seed);x,_=generate_noise_from_psd(n,np.array([0.,1.]),np.array([S,S]),fs)
  p=TimeSeries(x[0],sample_rate=fs).psd(fftlength=1024,overlap=512,method='welch');freq=np.asarray(p.frequencies);keep=(freq>.05)&(freq<.9);ratios.append(float(np.asarray(p)[keep].mean()/S))
  y=noise_from_one_sided_psd(np.full(n//2+1,S),fs,n,np.random.default_rng(seed));p=TimeSeries(y,sample_rate=fs).psd(fftlength=1024,overlap=512,method='welch');new.append(float(np.asarray(p)[keep].mean()/S))
 result['noise_normalization']={'legacy_measured_psd_over_requested':float(np.mean(ratios)),'legacy_expected':2/n,'corrected_measured_psd_over_requested':float(np.mean(new)),'actual_legacy_generation_n':2*(len(fi)-1),'actual_expected_psd_factor':2/(2*(len(fi)-1))}
 q=.6;eta=q/(1+q)**2;j=eta*(2*np.sqrt(3)-3.5171*eta+2.5763*eta**2);mass=1e6;freqs=[]
 for idx,(l,m) in enumerate([(2,2),(2,1),(3,3),(4,4)],1):
  old=(wf.f1[idx]+wf.f2[idx]*((1-j) if idx<3 else (m-j))**wf.f3[idx])/(mass*wf.TSUN*2*np.pi)
  fixed=(wf.f1[idx]+wf.f2[idx]*(1-j)**wf.f3[idx])/(mass*lal.MTSUN_SI*2*np.pi)
  p,tau=conversions.get_lm_f0tau(mass,j,l,m)
  vec=lal.CreateCOMPLEX16Vector(1);lalsimulation.SimIMREOBGenerateQNMFreqV2fromFinal(vec,mass,j,l,m,1);laf=float(vec.data[0].real/(2*np.pi))
  freqs.append({'mode':f'{l}{m}0','legacy_hz':float(old),'corrected_fit_hz':float(fixed),'pycbc_pykerr_hz':float(p),'lal_hz':laf,'legacy_relative_frequency_error':float(old/p-1),'corrected_fit_relative_frequency_error':float(fixed/p-1)})
 result['qnm_crosscheck']=freqs
 equal_eta=.25;result['legacy_equal_mass_odd_mode_coefficients']={'21':.060+.099*equal_eta,'33':.671+.157*equal_eta}
 # This reference uses detector-frame REMNANT mass; it makes no unrecorded
 # conversion from progenitor total mass or source-frame mass/redshift.
 qstar=2.;a22=.25*np.exp(-qstar/7.5);kwargs=dict(final_mass=1e6,final_spin=.68,distance=10000.,lmns=['221','211','331','441'],amp220=a22,amp210=.13*np.sqrt(qstar-1),amp330=.18*(qstar-1)**(1/3),amp440=.024*qstar**.75,phi220=0.,phi210=0.,phi330=0.,phi440=0.,inclination=np.pi/4,delta_t=.5,t_final=1000.,harmonics='spherical',taper=False)
 hp,hc=get_td_from_final_mass_spin(**kwargs);hp2,hc2=get_td_from_final_mass_spin(**{**kwargs,'distance':20000.})
 assert np.allclose(np.asarray(hp2),np.asarray(hp)/2,rtol=1e-12,atol=0)
 n=8192;h=np.zeros(n,complex);event=4000;h[event:event+len(hp)]=np.asarray(hp)+1j*np.asarray(hc)
 orbits=EqualArmlengthOrbits();orbits.configure(linear_interp_setup=True)
 response=pyResponseTDI(2.,n,order=25,tdi='2nd generation',tdi_chan='XYZ',orbits=orbits,force_backend='cpu')
 response.get_projections(h,1.,.4,t0=1000.);xyz=np.asarray(response.get_tdi_delays());aet=np.asarray(AET(*xyz))
 response.get_projections(2*h,1.,.4,t0=1000.);twice=np.asarray(AET(*response.get_tdi_delays()));linearity=float(np.linalg.norm(twice-2*aet)/np.linalg.norm(2*aet));assert linearity<1e-10
 fgrid=np.fft.rfftfreq(n,.5);sn=[];noise=[];rng=np.random.default_rng(2026100841)
 for sensitivity in [A2TDISens,E2TDISens,T2TDISens]:
  # Model explicitly selected, no empirical extrapolation or unlabelled defaults.
  psd=np.zeros_like(fgrid);psd[1:]=sensitivity.get_Sn(fgrid[1:],model='scirdv1');assert np.isfinite(psd).all() and (psd>=0).all();sn.append(psd);noise.append(noise_from_one_sided_psd(psd,2.,n,rng))
 sn=np.asarray(sn);noise=np.asarray(noise);mask=np.zeros_like(aet);mask[:,4200:4266]=1;observation=aet*(1-mask)+noise
 assert np.isfinite(aet).all() and np.linalg.norm(aet)>0
 np.savez_compressed(OUT/'lisa_tdi_reference.npz',hp=h.real,hc=h.imag,XYZ=xyz,AET=aet,noise_AET=noise,psd_AET=sn,frequency_hz=fgrid,mask=mask,observation=observation,dt_seconds=.5)
 result['reference_smoke_test']={'remnant_detector_mass_msun':1e6,'distance_mpc':10000.,'spin':.68,'tdi':'2nd generation','noise_model':'SciRDv1','units':'fractional frequency TDI signal and matching channel PSD [1/Hz]','AET_convention':['(Z-X)/sqrt(2)','(X-2Y+Z)/sqrt(6)','(X+Y+Z)/sqrt(3)'],'response_linearity_relative_error':linearity,'distance_doubling_halves_strain':True,'observation_exact_Gh_plus_n':bool(np.array_equal(observation,aet*(1-mask)+noise)),'limitations':'single-source smoke test; equal-arm orbits, ideal stationary instrument noise; no galactic foreground; boundary padding must be cropped for a production dataset; not training-ready; independent noise channels assume equal-arm symmetric noise'}
 # GWOSC is a terrestrial-data provenance check, never a LISA-noise supplier.
 try:
  from gwosc.datasets import event_detectors
  result['gwosc_provenance_check']={'event':'GW150914','detectors':sorted(event_detectors('GW150914')),'used_in_LISA_dataset':False}
 except Exception as e:result['gwosc_provenance_check']={'query_error':str(e),'used_in_LISA_dataset':False}
 (OUT/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()

