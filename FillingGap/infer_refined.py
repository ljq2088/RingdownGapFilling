"""Run refined DGF from legacy-benchmark observation and known gap mask only.
Input NPZ: observation [N,1056], mask [N,1056], mask=1 inside missing signal.
Additional fields in NPZ are ignored. Output NPZ: reconstruction [N,1056].
"""
from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from model.mask_aware_refiner import RefinedDGF
from model.QTranTimeMixerMod import QTransformModule
def validate(observation,mask):
 if observation.ndim!=2 or observation.shape[1]!=1056 or len(observation)==0:raise ValueError('observation must be nonempty [N,1056] legacy benchmark amplitudes')
 if mask.shape!=observation.shape or not np.isin(mask,[0,1]).all():raise ValueError('mask must have the same shape, with only 0/1 values; 1 means gap')
 if not np.isfinite(observation).all():raise ValueError('observation contains nonfinite values')
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
 parser.add_argument('--baseline',type=Path,default=ROOT/'saved_models/noise/lowSNR/model_with_noise_gaps_HugeRangeParas_14.pth')
 parser.add_argument('--refiner',type=Path,default=ROOT/'FillingGap/results_augmented_longrun_20261008/best_refiner.pth')
 parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');parser.add_argument('--batch-size',type=int,default=32);args=parser.parse_args()
 if args.batch_size<1:parser.error('--batch-size must be positive')
 if args.output.exists():raise FileExistsError('Output already exists; choose a new output path.')
 if args.output.resolve()==args.input.resolve():raise ValueError('Output cannot overwrite input')
 with np.load(args.input,allow_pickle=False) as z:
  observation=z['observation'].astype(np.float32);mask=z['mask'].astype(np.float32)
 validate(observation,mask);device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
 torch.set_num_threads(4);model=RefinedDGF.from_checkpoints(args.baseline,args.refiner,device);q=QTransformModule();outputs=[]
 with torch.no_grad():
  for lo in range(0,len(observation),args.batch_size):
   cpu=torch.from_numpy(observation[lo:lo+args.batch_size]);qt=torch.stack([q(row) for row in cpu]).float().to(device)
   y=model(cpu.to(device),qt,torch.from_numpy(mask[lo:lo+args.batch_size]).to(device)).cpu().numpy()
   if not np.isfinite(y).all():raise RuntimeError('Model produced nonfinite output')
   outputs.append(y)
 args.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.output,reconstruction=np.concatenate(outputs))
 print('Saved',args.output,'shape',observation.shape,flush=True)
if __name__=='__main__':main()
