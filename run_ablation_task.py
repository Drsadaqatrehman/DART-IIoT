import argparse, json, torch, pandas as pd
from pathlib import Path
from real_benchmarks import load_unsw, load_nbaiot, prepare_explicit_fold
from dart_iiot import ModelConfig, TrainConfig, CalibrationConfig, train_model, infer_raw, fit_calibrator, evaluate_variant
p=argparse.ArgumentParser(); p.add_argument('--dataset',choices=['unsw','nbaiot'],required=True); p.add_argument('--seed',type=int,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
torch.set_num_threads(4)
if a.dataset=='unsw':
 df=load_unsw(Path('/mnt/data/UNSW_NB15_training-set.csv'),Path('/mnt/data/UNSW_NB15_testing-set.csv'),5000,2500); held='Reconnaissance'; epochs=20; name='UNSW-NB15'
else:
 import real_benchmarks as rb; rb.NBAIOT_FAMILY_LEVEL=True
 df=load_nbaiot(Path('/mnt/data/DART_IIoT_real_experiments/data/nbaiot'),3000); held='Mirai'; epochs=15; name='N-BaIoT'
fold=prepare_explicit_fold(df,name,held,16,4,False,32)
tc=TrainConfig(epochs=epochs,batch_size=256,patience=6,lambda_rec=.05,lambda_proto=.01,lambda_sep=.005,focal_gamma=0.0)
model,h=train_model(fold,ModelConfig(),tc,a.seed,torch.device('cpu'))
rc=infer_raw(model,fold.X_cal,256,torch.device('cpu')); rt=infer_raw(model,fold.X_test,256,torch.device('cpu')); cal=fit_calibrator(rc,fold.y_cal,CalibrationConfig(alpha=.025))
variants=['classification_only','reconstruction_only','energy_only','prototype_only','without_reconstruction','without_energy','without_prototype','without_conformal','complete']
rows=[]
for v in variants:
 r=evaluate_variant(rt,fold.y_test,fold.unknown_test,cal,v,rc,fold.y_cal)
 rows.append({'dataset':name,'held_out_attack':held,'seed':a.seed,'variant':v,**{k:x for k,x in r.items() if k!='confusion_matrix'}})
a.out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(a.out,index=False)
if a.seed==1:
 pd.DataFrame({'epoch':range(1,len(h.train_loss)+1),'train_loss':h.train_loss,'val_loss':h.val_loss}).to_csv(a.out.with_name(a.out.stem+'_history.csv'),index=False)
print(pd.DataFrame(rows)[['variant','known_macro_f1','unknown_auroc','open_set_macro_f1']].to_string(index=False))
