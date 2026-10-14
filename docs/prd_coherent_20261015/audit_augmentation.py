from pathlib import Path
import sys,json,numpy as np,torch
ROOT=Path('/home/ljq/code/RingdownGapFilling_PRD');sys.path.insert(0,str(ROOT))
from prd_coherent.settings import OUT,CAL,CACHE,SEED
from prd_conditioned.data import MixedData
from prd_physical.common import Preprocess,write
from prd_variable.data import variable_mask
from prd_lowrank.common import legacy_features
from prd_frequency.features import features
torch.set_num_threads(2);count=512;pre=Preprocess(np.load(CAL)['psd'],'white');data=MixedData('train');h=data.target[:count].numpy();res=[]
for j in (0,1):
    u=data.views[j]['input'][:count].numpy();m=data.views[j]['mask'][:count].numpy();s=data.sign[j,:count].numpy()
    res.append(u-pre.encode(pre.decode(h*s[:,None,None])*(1-m[:,None])))
seeds=np.load(CACHE/'train_seeds.npy')[:count];theta=np.array([np.random.default_rng(int(s)+SEED).uniform(0,2*np.pi) for s in seeds],dtype='f4')[:,None,None]
mask=variable_mask(np.load(CACHE/'train_mask.npy',mmap_mode='r')[:count],seeds,421500000)
expected=pre.encode(pre.decode(h)*(1-mask[:,None]))+np.cos(theta)*res[0]+np.sin(theta)*res[1]
actual=np.load(OUT/'mixed_cache/input.npy',mmap_mode='r')[:count];error=float(abs(expected-actual).max());assert error<3e-4,error
assert np.array_equal(mask,np.load(OUT/'mixed_cache/mask.npy',mmap_mode='r')[:count])
q=legacy_features(actual[:8]).numpy();low=features(actual[:8]);qe=float(abs(q-np.load(OUT/'mixed_cache/q.npy',mmap_mode='r')[:8]).max());le=float(abs(low-np.load(OUT/'mixed_cache/q_low.npy',mmap_mode='r')[:8]).max());assert qe<1e-5 and le<1e-5,(qe,le)
mix=actual-pre.encode(pre.decode(h)*(1-mask[:,None]));flat=[x.reshape(-1).astype(float) for x in res];corr=float(np.corrcoef(flat)[0,1]);ratio=float(np.var(mix)/((np.var(res[0])+np.var(res[1]))/2));assert abs(corr)<.1 and .75<ratio<1.25,(corr,ratio)
r=dict(passed=True,training_sources_checked=count,all_feature_channels_recomputed_from_observation=True,input_consistency_max_error=error,legacy_feature_max_error=qe,physical_frequency_feature_max_error=le,independent_noise_empirical_correlation=corr,mixed_to_original_average_variance_ratio=ratio,gaussian_covariance_identity='cos(theta)^2 + sin(theta)^2 = 1 for independent equal-covariance noise',generation=0,test_read=False)
write(OUT/'augmentation_audit.json',r);print(json.dumps(r,indent=2))
