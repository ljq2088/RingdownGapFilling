from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
fig,ax=plt.subplots(figsize=(7.1,3.3));ax.set_xlim(0,10.3);ax.set_ylim(0,5);ax.axis('off')
def box(x,y,width,height,text,color='#E6EEF4'):
 ax.add_patch(FancyBboxPatch((x,y),width,height,boxstyle='round,pad=0.05',facecolor=color,edgecolor='#334455',lw=.7));ax.text(x+width/2,y+height/2,text,ha='center',va='center',fontsize=8)
def arrow(start,end): ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',lw=.8,color='#334455'))
box(.1,2,1.25,.9,'Observation d\n1056 samples')
box(1.9,3.25,2.1,1.15,'Complex Morlet CWT\nmodulus + phase\n2 x 16 x 264')
box(1.9,.65,2.1,1.15,'Normalize / db4 subbands\n8 x 1056\nPatch: 8 x 32 x 64')
box(4.5,3.25,2.1,1.15,'Temporal / scale mixing\nConvolution + projection\n8 x 32 x 64')
box(4.5,.65,2.1,1.15,'Add branch features\nEmbedding + Conv2d\n16 attention blocks')
box(7.1,.65,2.6,1.15,'MLP / patch map\nOverlap average / subband sum\nConv1d / restore scale')
box(7.1,3.25,2.6,1.15,'Residual refinement\nd, base prediction, mask, time\n25,489 trainable parameters','#F7E9D7')
box(7.1,2,2.6,.65,'Reconstructed waveform')
arrow((1.35,2.65),(1.9,3.65));arrow((1.35,2.2),(1.9,1.2));arrow((4,3.8),(4.5,3.8));arrow((4,1.2),(4.5,1.2));arrow((5.5,3.25),(5.5,1.85));arrow((6.6,1.2),(7.1,1.2))
# Route the base prediction into refinement without crossing the output node.
ax.plot([9.75,10.05,10.05],[1.2,1.2,3.8],color='#334455',lw=.8);arrow((10.05,3.8),(9.75,3.8))
arrow((8.4,3.2),(8.4,2.7))
ax.text(8.4,.2,'Frozen original DGF',ha='center',fontsize=8);ax.text(5,4.85,'Checkpoint-specific computation; target used only in training/evaluation',ha='center',fontsize=9)
fig.tight_layout();fig.savefig(a.output/'architecture_prd.pdf');fig.savefig(a.output/'architecture_prd.png',dpi=180)
