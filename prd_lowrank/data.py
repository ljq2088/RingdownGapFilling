import numpy as np, torch
from prd_physical.common import N, Preprocess, metric
from prd_strategy.latent import coordinates
from .settings import CACHE, CAL, RANK, VIEWS, BATCH

FIELDS = {'features': (2 * RANK + 6,), 'coefficients': (RANK,), 'scale': (1,), 'endpoint_dc': (2,), 'endpoint_ny': (2,), 'sign': ()}
PARITY = torch.tensor((-1.)**np.arange(N), dtype=torch.float32)

def packet(u, p, mask, basis, sign):
    values = coordinates(u, p, mask, basis)
    f = np.fft.rfft(np.asarray(p, float))
    values.pop('endpoint')
    values['endpoint_dc'] = (f[..., 0].real / N).astype('f4')
    values['endpoint_ny'] = (f[..., -1].real / N).astype('f4')
    values['sign'] = np.asarray(sign, 'f4')
    return values

def endpoint(values):
    return values['endpoint_dc'][..., None] + values['endpoint_ny'][..., None] * PARITY

def reconstruct(model, values, basis):
    c = model(values['features'], values['coefficients'], values['scale'])
    return (c @ basis.T).reshape(-1, 2, N) + endpoint(values)

class TrainingData:
    def __init__(self, cache, count, kind):
        self.count, self.kind = count, kind
        self.maps = []
        for j in range(VIEWS):
            self.maps.append({k: np.load(cache / f'view{j}' / (k + '.npy'), mmap_mode='r') for k in FIELDS})
        self.target = np.load(CACHE / 'train_target.npy', mmap_mode='r')
        self.mask = np.load(CACHE / 'train_mask.npy', mmap_mode='r')
    def batch(self, indices):
        indices = np.asarray(indices)
        source = indices % self.count
        view = indices // self.count if self.kind == 'noise_augmented' else np.zeros_like(source)
        values = {k: np.empty((len(indices),) + shape, dtype='f4') for k, shape in FIELDS.items()}
        for j in range(VIEWS):
            chosen = view == j
            for k in FIELDS:
                values[k][chosen] = self.maps[j][k][source[chosen]]
        values['target'] = np.asarray(self.target[source], 'f4') * values['sign'][:, None, None]
        values['mask'] = np.asarray(self.mask[source], 'f4')
        return {k: torch.from_numpy(v) for k, v in values.items()}

def validation_data(cache):
    values = {k: torch.from_numpy(np.load(cache / 'validation' / (k + '.npy'))) for k in FIELDS}
    values['target'] = torch.from_numpy(np.load(CACHE / 'validation_target.npy'))
    values['mask'] = torch.from_numpy(np.load(CACHE / 'validation_mask.npy'))
    return values

def predict(model, values, basis):
    model.eval()
    result = []
    with torch.no_grad():
        for lo in range(0, len(values['features']), BATCH):
            v = {k: a[lo:lo+BATCH] for k, a in values.items()}
            result.append(reconstruct(model, v, basis).numpy())
    return np.concatenate(result)

def validate(model, values, basis):
    cal = np.load(CAL)
    pre = Preprocess(cal['psd'], 'white')
    truth = np.load(CACHE / 'validation_truth.npy', mmap_mode='r').astype(float)
    p = pre.decode(predict(model, values, basis))
    return dict(score=float(metric(truth, p, cal['psd']).mean()), time=float(metric(truth, p).mean()), relative_l2=float(np.sqrt(((p-truth)**2).sum((1, 2)) / (truth*truth).sum((1, 2))).mean()))
