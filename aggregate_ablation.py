from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
root=Path('/mnt/data/DART_IIoT_real_experiments/ablation')
files=[p for p in root.glob('*.csv') if 'history' not in p.name]
frames=[pd.read_csv(p) for p in files]
raw=pd.concat(frames,ignore_index=True); raw.to_csv('/mnt/data/DART_IIoT_real_experiments/aggregate/real_ablation_per_run.csv',index=False)
metrics=['known_macro_f1','unknown_auroc','unknown_auprc','fpr_at_95_tpr','open_set_macro_f1','known_acceptance_rate','unknown_rejection_rate']
s=raw.groupby(['dataset','variant'])[metrics].agg(['mean','std','count']); s.columns=['_'.join(c) for c in s.columns]; s=s.reset_index(); s.to_csv('/mnt/data/DART_IIoT_real_experiments/aggregate/real_ablation_summary.csv',index=False)
labels={'classification_only':'Classification only','reconstruction_only':'Reconstruction only','energy_only':'Energy only','prototype_only':'Prototype only','without_reconstruction':'Without reconstruction','without_energy':'Without energy','without_prototype':'Without prototype','without_conformal':'Without conformal','complete':'Complete DART-IIoT'}
order=list(labels)
for metric,title,ylabel,fname in [('unknown_auroc','Real-data evidence ablation','Unknown AUROC','fig_real_ablation_auroc.pdf'),('open_set_macro_f1','Real-data decision ablation','Open-set macro-$F_1$','fig_real_ablation_openf1.pdf')]:
    p=s.pivot(index='variant',columns='dataset',values=metric+'_mean').reindex(order)
    e=s.pivot(index='variant',columns='dataset',values=metric+'_std').reindex(order)
    p.index=[labels[x] for x in p.index]
    ax=p.plot(kind='bar',yerr=e,figsize=(8.2,4.0),capsize=3,rot=28)
    ax.set_ylabel(ylabel); ax.set_xlabel(''); ax.set_title(title); ax.set_ylim(0,1.05); ax.grid(axis='y',alpha=.25); ax.legend(frameon=False)
    plt.tight_layout(); out=Path('/mnt/data/DART_IIoT_real_experiments/aggregate')/fname; plt.savefig(out,bbox_inches='tight'); plt.savefig(str(out).replace('.pdf','.png'),dpi=300,bbox_inches='tight'); plt.close()
print(s.to_string(index=False))
