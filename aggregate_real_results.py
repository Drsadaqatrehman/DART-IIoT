from pathlib import Path
import json, math
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

ROOT=Path('/mnt/data/DART_IIoT_real_experiments')
OUT=ROOT/'aggregate'; OUT.mkdir(parents=True,exist_ok=True)
files=sorted((ROOT/'runs').glob('*/per_run_metrics.csv'))
if not files:
    raise SystemExit('no results')
frames=[]
for f in files:
    df=pd.read_csv(f)
    df['run_dir']=f.parent.name
    frames.append(df)
raw=pd.concat(frames,ignore_index=True)
raw.to_csv(OUT/'real_per_run_metrics.csv',index=False)
metrics=['known_macro_f1','known_mcc','unknown_auroc','unknown_auprc','fpr_at_95_tpr','open_set_macro_f1','known_acceptance_rate','unknown_rejection_rate']
summary=raw.groupby(['dataset','method'])[metrics].agg(['mean','std','count'])
summary.columns=['_'.join(c) for c in summary.columns]
summary=summary.reset_index()
summary.to_csv(OUT/'real_baseline_summary.csv',index=False)
byfold=raw.groupby(['dataset','held_out_attack','method'])[metrics].agg(['mean','std','count'])
byfold.columns=['_'.join(c) for c in byfold.columns]
byfold=byfold.reset_index()
byfold.to_csv(OUT/'real_by_holdout_summary.csv',index=False)
# DART deployment
cols=['parameters','fp32_size_kib','estimated_int8_size_kib','latency_median_ms_batch1','latency_p95_ms_batch1']
dep=raw[raw.method=='DART-IIoT'].groupby('dataset')[cols].agg(['mean','std']).reset_index()
dep.columns=['dataset']+['_'.join(c) for c in dep.columns.tolist()[1:]]
dep.to_csv(OUT/'deployment_summary.csv',index=False)
# paired tests by dataset, aligned on holdout/seed
rows=[]
for dataset,g in raw.groupby('dataset'):
    dart=g[g.method=='DART-IIoT'].set_index(['held_out_attack','seed'])
    for method in sorted(set(g.method)-{'DART-IIoT'}):
        b=g[g.method==method].set_index(['held_out_attack','seed'])
        common=dart.index.intersection(b.index)
        for metric in ['known_macro_f1','unknown_auroc','open_set_macro_f1']:
            x=dart.loc[common,metric].astype(float); y=b.loc[common,metric].astype(float)
            try:
                stat,p=wilcoxon(x,y,zero_method='wilcox',alternative='two-sided')
            except Exception:
                stat,p=np.nan,np.nan
            diff=(x-y).to_numpy()
            rows.append({'dataset':dataset,'baseline':method,'metric':metric,'n':len(common),'dart_mean':x.mean(),'baseline_mean':y.mean(),'mean_difference':diff.mean(),'wilcoxon_stat':stat,'p_value':p})
pd.DataFrame(rows).to_csv(OUT/'paired_tests.csv',index=False)
# dataset record distributions from first corresponding run
for ds,prefix in [('UNSW-NB15','unsw_'),('N-BaIoT','nbaiot_')]:
    cand=sorted((ROOT/'runs').glob(prefix+'*/record_distribution.csv'))
    if cand:
        pd.read_csv(cand[0]).to_csv(OUT/(ds.replace('-','_')+'_record_distribution.csv'),index=False)

# Figures: one figure per metric, no custom colors/styles.
order=['Isolation Forest','MLP-MSP','GRU-MSP','CNN-GRU-MSP','DART-IIoT']
for metric,title,ylabel,fname in [
    ('known_macro_f1','Known-attack classification','Macro-$F_1$','fig_real_known_f1.pdf'),
    ('unknown_auroc','Unknown-attack ranking','AUROC','fig_real_unknown_auroc.pdf'),
    ('open_set_macro_f1','Three-way open-set decision','Open-set macro-$F_1$','fig_real_open_f1.pdf')]:
    pivot=summary.pivot(index='method',columns='dataset',values=metric+'_mean').reindex(order)
    err=summary.pivot(index='method',columns='dataset',values=metric+'_std').reindex(order)
    ax=pivot.plot(kind='bar',yerr=err,figsize=(7.2,3.8),capsize=3,rot=20)
    ax.set_ylabel(ylabel); ax.set_xlabel(''); ax.set_title(title); ax.set_ylim(0,1.05); ax.grid(axis='y',alpha=.25)
    ax.legend(frameon=False,ncol=2)
    plt.tight_layout(); plt.savefig(OUT/fname,bbox_inches='tight'); plt.savefig(OUT/fname.replace('.pdf','.png'),dpi=300,bbox_inches='tight'); plt.close()

# DART by holdout
D=byfold[byfold.method=='DART-IIoT'].copy()
D['label']=D['dataset']+' / '+D['held_out_attack']
for metric,title,ylabel,fname in [
    ('unknown_auroc','DART-IIoT unknown-family performance','Unknown AUROC','fig_dart_by_holdout_auroc.pdf'),
    ('open_set_macro_f1','DART-IIoT three-way decision performance','Open-set macro-$F_1$','fig_dart_by_holdout_openf1.pdf')]:
    fig,ax=plt.subplots(figsize=(7.2,3.8))
    x=np.arange(len(D)); ax.bar(x,D[metric+'_mean'],yerr=D[metric+'_std'],capsize=3)
    ax.set_xticks(x,D['label'],rotation=25,ha='right'); ax.set_ylabel(ylabel); ax.set_ylim(0,1.05); ax.set_title(title); ax.grid(axis='y',alpha=.25)
    plt.tight_layout(); plt.savefig(OUT/fname,bbox_inches='tight'); plt.savefig(OUT/fname.replace('.pdf','.png'),dpi=300,bbox_inches='tight'); plt.close()

print('files',len(files),'rows',len(raw))
print(summary.to_string(index=False))
