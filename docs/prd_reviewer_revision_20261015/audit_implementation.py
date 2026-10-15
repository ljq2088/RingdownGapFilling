import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['SSQ_GPU']='0'
from pathlib import Path
import json,hashlib,importlib.metadata as md
import numpy as np,torch
P=Path(__file__).resolve().parent;R=Path('/home/ljq/code/RingdownGapFilling_PRD')
versions={}
for name in ('pycbc','lalsuite','fastlisaresponse','lisa-analysis-tools','ssqueezepy','PyWavelets','torch','numpy','scipy','astropy'):
    try:versions[name]=md.version(name)
    except md.PackageNotFoundError:versions[name]=None
from ssqueezepy import Wavelet
wavelet=Wavelet('morlet')
paths={
 'legacy_initialization':'saved_models/noise/lowSNR/model_with_noise_gaps_HugeRangeParas_14.pth',
 'physical_transfer':'FillingGap/results_prd_physical_transfer_20261009/white/best.pth',
 'joint_channel':'FillingGap/results_prd_optimization_20261010/joint_hann/best.pth',
 'frequency_adapter':'FillingGap/results_prd_strategy_20261013/frequency_snapshot.pth',
 'basis':'FillingGap/results_prd_strategy_20261013/training_direction_basis.npz',
 'coefficient_adapter':'FillingGap/results_prd_lowrank_20261013/noise_augmented/best.pth',
 'reference':'FillingGap/results_prd_variable_20261013/joint_finetune/best.pth',
 'conditioned_head':'FillingGap/results_prd_conditioned_20261014/head_full/best.pth',
 'final':'FillingGap/results_prd_conditioned_20261014/finetune/best.pth',
 'psd':'FillingGap/results_prd_physical_transfer_20261009/calibration_psd.npz',
 'noise_rms':'FillingGap/results_prd_conditioned_20261014/noise_calibration.npz',
}
assets={}
for key,name in paths.items():
    p=R/name
    assert p.exists(),p
    assets[key]=dict(path=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    if p.suffix=='.pth':
        x=torch.load(p,map_location='cpu',weights_only=False)
        assets[key]['metadata']={k:v for k,v in x.items() if k in ('epoch','score','kind','variant','weight_type','rank','count')}
noise=np.load(R/paths['noise_rms'])
assets['noise_rms']['parameters']={k:noise[k].tolist() for k in noise.files}
selection=json.loads((R/'FillingGap/results_prd_conditioned_20261014/pilot_selection.json').read_text())
seeds={}
cache=R/'FillingGap/results_prd_optimization_20261013/cache'
for split in ('train','validation'):
    x=np.load(cache/(split+'_seeds.npy'));seeds[split]=dict(n=len(x),first=int(x[0]),last=int(x[-1]))
metadata=dict(software_versions=versions,morlet=dict(config=wavelet.config,cwt_scales=list(range(1,17)),padtype='reflect',l1_norm=True),assets=assets,source_seeds=seeds,pilot_selection=selection,observation='G*h+n',noise_rms_calibration='128 separate independent noise realizations; root mean of per-segment mean squares',phase_plot='unwrap(arg(z_target)) + arg(z_prediction*conj(z_target)) for predicted display',periodogram=dict(window='boxcar',detrend='constant',scaling='density',onesided=True,aggregation='arithmetic mean of individual periodograms'),metric='direct signed joint A/E PSD-weighted mismatch; excludes DC and Nyquist')
(P/'implementation_audit.json').write_text(json.dumps(metadata,indent=2,default=str))
print(json.dumps(dict(software_versions=versions,morlet=metadata['morlet'],noise_rms=assets['noise_rms']['parameters'],source_seeds=seeds),indent=2,default=str))
