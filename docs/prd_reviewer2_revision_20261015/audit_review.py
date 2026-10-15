"""Read-only CPU audit of the substantive second-review questions."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['SSQ_GPU']='0'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import sys,json,hashlib,inspect
import numpy as np
import scipy.fft as fft
import torch
from ssqueezepy import cwt,Wavelet
from ssqueezepy.utils import padsignal
from threadpoolctl import threadpool_limits
threadpool_limits(2);torch.set_num_threads(2)
R=Path('/home/ljq/code/RingdownGapFilling_PRD');sys.path.insert(0,str(R))
P=Path(__file__).resolve().parent
C=R/'FillingGap/results_prd_optimization_20261013/cache'
F=R/'FillingGap/results_prd_conditioned_20261014'
N=1056;fs=2.
out={}
xp,_,nleft,nright=padsignal(np.zeros(N,np.float32),'reflect',get_params=True)
M=len(xp);w=Wavelet('morlet',N=M)
scales=np.arange(1,17,dtype=np.float32)
filters=np.asarray(w(scale=scales[:,None],nohalf=False))
freq=fft.fftfreq(M,1/fs);freq[M//2]=fs/2
rows=[]
for a,h in zip(scales,filters):
    k=int(np.argmax(abs(h)));pos=(freq>=0)
    rows.append(dict(scale=int(a),nominal_center_hz=float(13.4*fs/(2*np.pi*a)),discrete_peak_hz=float(freq[k]),peak_gain=float(abs(h[k])),response_energy=float(np.sum(abs(h.astype('complex128'))**2))))
lowfreq=np.geomspace(.25,.002,16);lowscales=(13.4*fs/(2*np.pi*lowfreq)).astype(np.float32)
lf=np.asarray(w(scale=lowscales[:,None],nohalf=False))
out['cwt']=dict(version='0.6.5',mu=13.4,input_samples=N,padded_samples=M,left_pad=int(nleft),right_pad=int(nright),rows=rows,low_nominal_hz=lowfreq.tolist(),low_discrete_peak_hz=[float(freq[int(np.argmax(abs(h)))]) for h in lf])
# Verify direct FFT filtering reproduces the installed CWT, including reflection.
x=np.random.default_rng(20261015).normal(size=N).astype(np.float32)
xpad,_,left,_=padsignal(x,'reflect',get_params=True)
expected=fft.ifft(filters*fft.fft(xpad),axis=-1)[:,left:left+N]
got,returned=cwt(x,'morlet',scales=scales)
out['cwt']['direct_filter_relative_error']=float(np.linalg.norm(got-expected)/np.linalg.norm(got))
assert out['cwt']['direct_filter_relative_error']<1e-6
psd=np.load(R/'FillingGap/results_prd_physical_transfer_20261009/calibration_psd.npz')['psd']
asd=np.sqrt(psd*fs/2);asd[:,0]=asd[:,1];asd[:,-1]=asd[:,-2]
signal=np.load(F/'fresh_test/truth.npy',mmap_mode='r')
energy=np.zeros(3);low_energy=[]
for lo in range(0,len(signal),96):
    v=np.fft.irfft(np.fft.rfft(np.asarray(signal[lo:lo+96],float))/asd,n=N)
    power=abs(np.fft.rfft(v))**2;power[...,0]=power[...,-1]=0
    f=np.fft.rfftfreq(N,1/fs)
    energy+=np.array([power.sum(),power[...,f<=.25].sum(),power[...,f>=13.4*fs/(2*np.pi*16)].sum()])
out['cwt']['test_whitened_target_energy_fractions']=dict(below_equal_025=float(energy[1]/energy[0]),above_equal_lowest_legacy_center=float(energy[2]/energy[0]),definition='ratio of population-summed interior-bin whitened target power; not an ablation')
# All source populations, including every training source; no outcome exclusions.
splits={}
for split in ('train','validation','test'):
    path=F/'fresh_test' if split=='test' else C
    pars=np.load(path/('parameters.npy' if split=='test' else split+'_parameters.npy'),mmap_mode='r')
    seeds=np.load(path/('seeds.npy' if split=='test' else split+'_seeds.npy'))
    original=np.load(path/('mask.npy' if split=='test' else split+'_mask.npy'),mmap_mode='r')
    peak=np.asarray(pars[:,13],int)
    delta=np.array([np.random.default_rng(int(s)+200000000).integers(-132,133) for s in seeds])
    local=np.array([np.random.default_rng(int(s)+300000000).integers(132,264) for s in seeds])
    g=np.clip(peak+delta,0,32768-66);origin=np.clip(g-local,0,32768-N);a=g-origin
    assert np.array_equal(a,np.asarray(original).argmax(-1))
    assert np.all(np.asarray(original).sum(-1)==66)
    valid=(a[:,None]-(np.arange(64,105,2)-66)//2>=0)&(a[:,None]-(np.arange(64,105,2)-66)//2+np.arange(64,105,2)<=N)
    assert valid.all()
    splits[split]=dict(n=len(seeds),parent_peak_min=int(peak.min()),parent_peak_max=int(peak.max()),clipped_parent_gap=int(np.sum(g!=peak+delta)),clipped_crop_origin=int(np.sum(origin!=g-local)),reference_start_min=int(a.min()),reference_start_max=int(a.max()),all_21_lengths_in_bounds=True,minimum_possible_start=int((a.min()-19)),maximum_possible_end=int(a.max()+85))
out['gap_boundaries']=splits
# Reproduce the noise calibration through its actual simulator, entirely on CPU.
from prd_physical.physics import Simulator
from prd_physical.common import Preprocess
sim=Simulator();pre=Preprocess(psd,'white');sq=[];periodograms=[]
for seed in range(2027100000,2027100128):
    noise=sim.noise(seed)[:,6000:6000+N]
    u=pre.encode(noise);f=np.fft.rfft(u);f[:,0]=f[:,-1]=0
    sq.append(np.mean(np.fft.irfft(f,n=N)**2,axis=-1))
    spec=abs(np.fft.rfft(noise))**2/(fs*N);spec[:,1:-1]*=2;periodograms.append(spec)
    if seed%32==0:print('noise calibration',seed,flush=True)
rms=np.sqrt(np.mean(sq,axis=0));stored=np.load(F/'noise_calibration.npz')
assert np.array_equal(rms,stored['rms'])
cal=np.load(R/'FillingGap/results_prd_physical_transfer_20261009/calibration_psd.npz')
out['noise_rms']=dict(recomputed=rms.tolist(),stored=stored['rms'].tolist(),bitwise_equal=True,psd_seed_start=int(cal['seed_start']),psd_segments=int(cal['n_noise']),rms_seed_range=[2027100000,2027100127],window_start=6000,relationship='first 128 of the same 512 independent noise realizations; regenerated deterministically for RMS; shared between estimators, separate from every source-data population')
# Inspect legacy data without running its generation or training code.
legacy=Path('/home/ljq/code/RingdownGapFilling/dataset/train_dataset_signal_data_noise_gaps_HugeRangeParas.pth')
bank=torch.load(legacy,map_location='cpu',weights_only=False)
conditions=bank.conditions.detach().cpu().numpy().reshape(len(bank),3)
physical=np.concatenate([np.asarray(np.load(C/(s+'_parameters.npy'),mmap_mode='r'))[:,:3] for s in ('train','validation')]+[np.asarray(np.load(F/'fresh_test/parameters.npy',mmap_mode='r'))[:,:3]])
# Float32 casting makes this an intentionally conservative exact-tuple screen.
legacy_tuples=set(map(tuple,conditions.astype('f4').tolist()))
overlap=sum(tuple(row) in legacy_tuples for row in physical.astype('f4').tolist())
out['legacy']=dict(n=len(bank),parameter_min=conditions.min(0).tolist(),parameter_max=conditions.max(0).tolist(),parameter_quantiles=np.quantile(conditions,[.01,.5,.99],axis=0).tolist(),condition_columns=['source_total_mass_msun','mass_ratio','redshift'],exact_parameter_tuple_overlap_with_current_populations=overlap,legacy_sha256=hashlib.sha256(legacy.read_bytes()).hexdigest(),training_source_loss='mean((prediction-target)^2 + abs(prediction-target)); configured mask weight alpha=1; supplied training code uses default l1_weight=1',historical_checkpoint_has_optimizer_or_split_metadata=False)
(P/'technical_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps(out,indent=2),flush=True)
