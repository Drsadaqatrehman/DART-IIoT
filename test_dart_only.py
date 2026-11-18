import torch
from pathlib import Path
from real_benchmarks import load_unsw, prepare_explicit_fold
from dart_iiot import ModelConfig, TrainConfig, CalibrationConfig, train_model, infer_raw, fit_calibrator, evaluate_variant

torch.set_num_threads(4)
train=Path('/mnt/data/UNSW_NB15_training-set.csv'); test=Path('/mnt/data/UNSW_NB15_testing-set.csv')
df=load_unsw(train,test,2500,1250)
fold=prepare_explicit_fold(df,'UNSW-NB15','DoS',16,8,False,32)
for e in [8,12]:
 print('start',e,flush=True)
 model,h=train_model(fold,ModelConfig(),TrainConfig(epochs=e,batch_size=256,patience=5),1,torch.device('cpu'))
 rc=infer_raw(model,fold.X_cal,256,torch.device('cpu')); rt=infer_raw(model,fold.X_test,256,torch.device('cpu'))
 cal=fit_calibrator(rc,fold.y_cal,CalibrationConfig(alpha=.025))
 r=evaluate_variant(rt,fold.y_test,fold.unknown_test,cal,'complete',rc,fold.y_cal)
 print('E',e,'best',h.best_epoch,'sec',h.seconds,{k:r[k] for k in ['known_macro_f1','unknown_auroc','unknown_auprc','fpr_at_95_tpr','open_set_macro_f1','unknown_rejection_rate','known_acceptance_rate']},flush=True)
