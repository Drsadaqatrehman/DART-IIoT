import torch, time
from pathlib import Path
from real_benchmarks import load_unsw, prepare_explicit_fold
from dart_iiot import ModelConfig, TrainConfig, CalibrationConfig, train_model, infer_raw, fit_calibrator, evaluate_variant

torch.set_num_threads(4)
df=load_unsw(Path('/mnt/data/UNSW_NB15_training-set.csv'),Path('/mnt/data/UNSW_NB15_testing-set.csv'),5000,2500)
fold=prepare_explicit_fold(df,'UNSW-NB15','DoS',16,4,False,32)
configs=[
 ('orig',TrainConfig(epochs=16,batch_size=256,patience=5)),
 ('balanced',TrainConfig(epochs=20,batch_size=256,patience=6,lambda_rec=.1,lambda_proto=.02,lambda_sep=.01,focal_gamma=1.0)),
 ('clsheavy',TrainConfig(epochs=20,batch_size=256,patience=6,lambda_rec=.05,lambda_proto=.01,lambda_sep=.005,focal_gamma=0.0)),
]
for name,tc in configs:
 print('START',name,flush=True)
 m,h=train_model(fold,ModelConfig(),tc,1,torch.device('cpu'))
 rc=infer_raw(m,fold.X_cal,256,torch.device('cpu'));rt=infer_raw(m,fold.X_test,256,torch.device('cpu'))
 cal=fit_calibrator(rc,fold.y_cal,CalibrationConfig(alpha=.025))
 for v in ['classification_only','complete']:
  r=evaluate_variant(rt,fold.y_test,fold.unknown_test,cal,v,rc,fold.y_cal)
  print(name,v,'best',h.best_epoch,{k:round(float(r[k]),4) for k in ['known_macro_f1','unknown_auroc','unknown_auprc','open_set_macro_f1','unknown_rejection_rate','known_acceptance_rate']},'weights',cal.fusion.weights,flush=True)
