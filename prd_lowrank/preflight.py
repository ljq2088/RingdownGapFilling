import time, json, copy
import numpy as np, torch
from prd_physical.common import Preprocess, metric, write, sha, N, FS
from prd_physical.physics import Simulator
from prd_physical.prepare import crop_meta
from prd_strategy.latent import CoefficientAdapter, WholeWindowAdapter
from prd_strategy.loss import StrategyObjective
from prd_optimize.loss import Objective
from .settings import *
from .common import status, inferred, load_backbone
from .prepare import main as prepare
from .train import train_variant
from .data import TrainingData, validation_data, reconstruct, validate, packet, predict

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    # Existing running experiment stays bitwise frozen.
    manifest = json.loads((CACHE.parent/'runtime_source_manifest.json').read_text())
    assert all(sha(ROOT/name)==expected for name,expected in manifest.items())
    assert sha(PROBE/'frequency_snapshot.pth') == SNAPSHOT_SHA
    assert json.loads((PROBE/'completion_gates.json').read_text())['passed']
    tr = np.load(CACHE/'train_seeds.npy');va = np.load(CACHE/'validation_seeds.npy')
    test = np.arange(TEST_SEED, TEST_SEED+TEST_N)
    assert len(tr)==TRAIN_N and len(va)==VAL_N and not np.isin(tr,va).any()
    assert not np.isin(test,np.r_[tr,va]).any()
    assert TEST_SEED != __import__('prd_frequency.settings',fromlist=['TEST_SEED']).TEST_SEED
    cal = np.load(CAL);pre = Preprocess(cal['psd'],'white')
    sim = Simulator();sums = np.zeros_like(cal['psd']);stds=[]
    params = np.load(CACHE/'train_parameters.npy',mmap_mode='r');t=time.monotonic()
    for i in range(512):
        start,_ = crop_meta(params[i],int(tr[i]))
        n = sim.noise(NOISE_SEED+i)[:,start:start+N].astype('f4')
        period = np.abs(np.fft.rfft(n.astype(float)))**2/(FS*N);period[:,1:-1]*=2;sums += period
        stds.append(pre.encode(n).std(-1))
    ratio = (sums/512/cal['psd'])[:,1:-1]
    median = np.median(ratio,axis=-1)
    assert np.all((median>.75)&(median<1.35)), median
    sample = OUT/'preflight_run'
    prepare(sample,128)
    data = TrainingData(sample/'cache',128,'noise_augmented')
    basis64 = np.load(PROBE/'training_direction_basis.npz')['vectors'][:,:RANK]
    basis = torch.tensor(basis64,dtype=torch.float32)
    val = validation_data(sample/'cache');model=CoefficientAdapter(RANK)
    initial=validate(model,val,basis)
    v=data.batch(np.arange(128));y=reconstruct(model,v,basis)
    legacy=StrategyObjective(pre,cal,False)(y,v['target'],v['mask'])
    original=Objective(pre,cal,'joint_hann',device='cpu')(y,v['target'],v['mask'])
    assert abs(float((legacy-original).detach()))<1e-6
    loss=StrategyObjective(pre,cal,True)(y,v['target'],v['mask']);loss.backward()
    gradient=float(model.head.weight.grad.norm());assert gradient>0
    u=np.load(CACHE/'train_input.npy',mmap_mode='r')[:16];m=np.load(CACHE/'train_mask.npy',mmap_mode='r')[:16]
    backbone=load_backbone();current=inferred(backbone,u,m);known=np.load(PROBE/'frequency_snapshot_train_prediction.npy')[:16]
    feature_relative=float(np.linalg.norm(current-known)/np.linalg.norm(known));assert feature_relative<1e-6
    # Compact endpoints reproduce the existing deployed rank-128 candidate.
    ck=torch.load(PROBE/'head_balanced/best.pth',map_location='cpu',weights_only=True)
    model.load_state_dict(ck['state_dict'])
    got=predict(model,val,basis);expected=np.load(PROBE/'head_balanced/validation_prediction.npy')
    rel=float(np.linalg.norm(got-expected)/np.linalg.norm(expected));assert rel<1e-6
    assert abs(validate(model,val,basis)['score']-ck['score'])<1e-8
    # Interrupted and continuous training have identical states and history scores.
    import shutil
    resume=OUT/'resume_gate';continuous=OUT/'continuous_gate'
    for dst in [resume,continuous]:
        dst.mkdir(exist_ok=True)
        if not (dst/'cache').exists():(dst/'cache').symlink_to(sample/'cache',target_is_directory=True)
    train_variant('noise_augmented',resume,128,3,1,100,stop_after=1)
    train_variant('noise_augmented',resume,128,3,1,100)
    train_variant('noise_augmented',continuous,128,3,1,100)
    a=torch.load(resume/'noise_augmented/last.pth',map_location='cpu',weights_only=False)
    b=torch.load(continuous/'noise_augmented/last.pth',map_location='cpu',weights_only=False)
    for key in ['raw','ema','best_state']:
        assert all(torch.equal(a[key][k],b[key][k]) for k in a[key]),key
    assert torch.equal(a['shuffle_rng'],b['shuffle_rng'])
    assert [x['validation'] for x in a['history']]==[x['validation'] for x in b['history']]
    # Compute-only throughput benchmark: its weights are discarded.
    torch.manual_seed(SEED);bench=CoefficientAdapter(RANK);opt=torch.optim.AdamW(bench.parameters(),lr=2e-4);objective=StrategyObjective(pre,cal,True)
    t=time.monotonic()
    for i in range(64):
        v=data.batch((np.arange(BATCH)+i*BATCH)%(128*VIEWS));opt.zero_grad(set_to_none=True)
        loss=objective(reconstruct(bench,v,basis),v['target'],v['mask']);loss.backward();opt.step()
    seconds=(time.monotonic()-t)/64
    result=dict(passed=True,train_unique=TRAIN_N,validation_unique=VAL_N,test_unique=TEST_N,new_noise_psd_ratio_channel_medians=median.tolist(),whitened_noise_std_channel_means=np.mean(stds,axis=0).tolist(),initial_projection_validation=initial,old_objective_max_error=abs(float((legacy-original).detach())),head_gradient_norm=gradient,pilot_prediction_relative_error=rel,resume_bitwise_equal=True,recomputed_feature_prediction_relative_error=feature_relative,train_validation_test_seed_ranges_disjoint=True,current_long_sources_unchanged=True,test_read=False,compute_only_seconds_per_batch=seconds,estimated_full_epoch_seconds=seconds*np.ceil(TRAIN_N*VIEWS/BATCH),benchmark_weights_discarded=True)
    write(OUT/'preflight.json',result);status(OUT,'preflight_passed',**result)
if __name__=='__main__':main()
