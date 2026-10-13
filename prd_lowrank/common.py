import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '2')
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
import json, time
from pathlib import Path
import numpy as np, torch
from prd_physical.common import write, sha, N
from .settings import ROOT, OUT, MIN_RAM_GIB

def status(path, stage, **values):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    value = dict(stage=stage, pid=os.getpid(), updated_unix=time.time(), **values)
    write(path / 'status.json', value)
    print(json.dumps(value), flush=True)

def check_sources():
    manifest = OUT / 'runtime_source_manifest.json'
    if manifest.exists():
        for name, expected in json.loads(manifest.read_text()).items():
            if sha(ROOT / name) != expected:
                raise RuntimeError('frozen runtime changed: ' + name)
    assets = OUT / 'frozen_asset_manifest.json'
    if assets.exists():
        for name, expected in json.loads(assets.read_text()).items():
            if sha(ROOT / name) != expected:
                raise RuntimeError('frozen asset changed: ' + name)

def ram_guard(path):
    import psutil
    while psutil.virtual_memory().available < MIN_RAM_GIB * 2**30:
        status(path, 'waiting_for_ram', available_gib=psutil.virtual_memory().available / 2**30)
        time.sleep(5)

def atomic_checkpoint(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp.pth')
    torch.save(value, tmp)
    tmp.replace(path)

def legacy_features(u):
    from ssqueezepy import cwt
    scale = np.sqrt(np.mean(np.asarray(u, float)**2, axis=-1)).clip(1e-8)
    normalized = (u / scale.astype('f4')[..., None]).reshape(-1, N)
    z, _ = cwt(normalized, 'morlet', scales=np.linspace(1, 16, 16))
    z = torch.from_numpy(z.astype('complex64'))
    return torch.stack([z.abs(), z.angle()], dim=1)[..., ::4].reshape(len(u), 2, 2, 16, 264)

def inferred(model, u, mask):
    from prd_frequency.features import features
    x = torch.from_numpy(np.array(u, copy=True))
    m = torch.from_numpy(np.array(mask, copy=True))
    q = legacy_features(u)
    low = torch.from_numpy(features(u))
    with torch.inference_mode():
        return model(x, q, m, model.input_bands(x), low).numpy()

def load_backbone():
    from prd_frequency.model import FrequencyDGF
    from .settings import PROBE, SNAPSHOT_SHA
    assert sha(PROBE / 'frequency_snapshot.pth') == SNAPSHOT_SHA
    model = FrequencyDGF('frequency_adapter').cpu().eval()
    model.load_state_dict(torch.load(PROBE / 'frequency_snapshot.pth', map_location='cpu', weights_only=True)['state_dict'])
    return model
