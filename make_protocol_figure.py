import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fig,ax=plt.subplots(figsize=(10.5,3.0)); ax.set_xlim(0,10.5); ax.set_ylim(0,3); ax.axis('off')
boxes=[
(0.2,1.05,1.7,1.0,'Datasets','UNSW-NB15\nN-BaIoT\nControlled stream'),
(2.2,1.05,1.7,1.0,'Preprocessing','Robust scaling\nTraining-only selection\n32 features'),
(4.2,1.05,1.7,1.0,'Temporal input','16-record windows\nStride 4\nNo split crossing'),
(6.2,1.05,1.7,1.0,'Open-set protocol','Held-out family\nTrain / calibration / test\nThree seeds'),
(8.2,1.05,2.0,1.0,'Evaluation','DART-IIoT + baselines\nKnown and unknown metrics\nLatency and ablation')]
for x,y,w,h,title,text in boxes:
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.03,rounding_size=0.08',linewidth=1.2,facecolor='white'))
    ax.text(x+w/2,y+h-0.22,title,ha='center',va='center',fontweight='bold',fontsize=10)
    ax.text(x+w/2,y+0.38,text,ha='center',va='center',fontsize=8.4,linespacing=1.25)
for i in range(len(boxes)-1):
    x1=boxes[i][0]+boxes[i][2]; x2=boxes[i+1][0]
    ax.add_patch(FancyArrowPatch((x1+0.04,1.55),(x2-0.04,1.55),arrowstyle='-|>',mutation_scale=12,linewidth=1.2))
ax.text(5.25,2.55,'Leakage-resistant experimental protocol',ha='center',va='center',fontweight='bold',fontsize=12)
plt.tight_layout();
out='/mnt/data/DART_IIoT_real_experiments/aggregate/fig_experimental_protocol.pdf'; plt.savefig(out,bbox_inches='tight'); plt.savefig(out.replace('.pdf','.png'),dpi=300,bbox_inches='tight')
