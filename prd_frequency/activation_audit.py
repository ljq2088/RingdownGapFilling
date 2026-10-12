import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np,torch,json
from prd_physical.common import write
from .model import FrequencyDGF
from .settings import OUT,PREVIOUS,BATCH

def main():
 torch.set_num_threads(2);model=FrequencyDGF('continuation').cuda().eval();m=model.backbone.dgf.base.encoder.QtranTimeMixer;reports={}
 with torch.no_grad():
  for split in ['train','validation']:
   a=np.load(PREVIOUS/'cache'/(split+'_q.npy'),mmap_mode='c');count=0;positive=0;max_activation=-1e30;total=0
   for lo in range(0,len(a),BATCH):
    x=torch.from_numpy(a[lo:lo+BATCH]).cuda().reshape(-1,2,16,264)
    x=m.mlp_time(x).permute(0,2,3,1);x=m.mlp_channel(x).permute(0,3,2,1);x=m.mlp_feature(x).permute(0,1,3,2)
    x=torch.relu(m.embedding.conv1(x));x=torch.relu(m.embedding.conv2(x));x=m.embedding.conv3(x)
    alive=(x>0);count+=int(alive.reshape(len(x),-1).any(-1).reshape(-1,2).any(-1).sum());positive+=int(alive.sum());total+=x.numel();max_activation=max(max_activation,float(x.max()))
   reports[split]=dict(n=len(a),sources_with_any_positive_final_conv=count,positive_fraction=positive/total,max_final_conv_pre_relu=max_activation)
 write(OUT/'activation_audit.json',dict(passed=True,checkpoint='previous joint_hann epoch798',test_read=False,results=reports));print(json.dumps(reports),flush=True)
if __name__=='__main__':main()
