"""Scientific checks on normalization, direct phase sensitivity, and PSD weights."""
import importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('evaluation',Path(__file__).with_name('evaluate_prd_revision.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
f=module.overlap;t=np.arange(1056);phase=.3
h=np.cos(2*np.pi*7*t/1056)[None,:]
p=np.cos(2*np.pi*7*t/1056+phase)[None,:]
assert np.allclose(f(h,p),1-np.cos(phase),atol=1e-12)
assert np.allclose(f(h,3*h),0,atol=1e-12)
assert np.allclose(f(h,-h),2,atol=1e-12)
w=np.ones(529);w[[0,-1]]=0
assert np.allclose(f(h,p,w),f(h,p),atol=1e-12)
assert np.allclose(f(h,p,w*1e20),f(h,p,w),atol=1e-12)
p=h+.5*np.cos(2*np.pi*19*t/1056)[None,:]
w2=w.copy();w2[19]=100
assert f(h,p,w2)[0]>f(h,p,w)[0]
# PSD -> 0 in a bin carrying prediction error increases its penalty;
# it does not force a normalized match to approach one.
print('PASS: direct phase, amplitude invariance, signed range, Parseval agreement, PSD scale invariance, low-PSD error penalty.')
