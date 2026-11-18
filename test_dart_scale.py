import torch
from pathlib import Path
from real_benchmarks import load_unsw, prepare_explicit_fold
from dart_iiot import ModelConfig, TrainConfig, CalibrationConfig, train_model, infer_raw, fit_calibrator, evaluate_variant

torch.set_num_threads(4)
df=load_unsw(Path('/mnt/data/UNSW_NB15_training-set.csv'),Path('/mnt/data/UNSW_NB15_testing-set.csv'),5000,2500)
for stride in [4,2]:
 fold=prepare_explicit_fold(df,'UNSW-NB15','DoS',16,stride,False,32)
 print('stride',stride,fold.metadata,flush=True)
 model,h=train_model(fold,ModelConfig(conv_channels=48,gru_hidden=64,latent_dim=48),TrainConfig(epochs=25,batch_size=256,patience=6,learning_rate=1e-3),1,torch.device('cpu'))
 rc=infer_raw(model,fold.X_cal,256,torch.device('cpu')); rt=infer_raw(model,fold.X_test,256,torch.device('cpu'))
 cal=fit_calibrator(rc,fold.y_cal,CalibrationConfig(alpha=.05,confidence_quantile=.001,prototype_quantile=.999))
 for v in ['classification_only','energy_only','without_conformal','complete']:
  r=evaluate_variant(rt,fold.y_test,fold.unknown_test,cal,v,rc,fold.y_cal)
  print(v,{k:r[k] for k in ['known_macro_f1','unknown_auroc','unknown_auprc','fpr_at_95_tpr','open_set_macro_f1','unknown_rejection_rate','known_acceptance_rate']},flush=True)
