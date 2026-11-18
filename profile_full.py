import time, torch
from pathlib import Path
from real_benchmarks import load_unsw, prepare_explicit_fold
from dart_iiot import ModelConfig, TrainConfig, CalibrationConfig, train_model, infer_raw, fit_calibrator, evaluate_variant

torch.set_num_threads(4)
df=load_unsw(Path('/mnt/data/UNSW_NB15_training-set.csv'),Path('/mnt/data/UNSW_NB15_testing-set.csv'),5000,2500)
fold=prepare_explicit_fold(df,'UNSW-NB15','DoS',16,4,False,32)
t=time.time();m,h=train_model(fold,ModelConfig(),TrainConfig(epochs=12,batch_size=256,patience=5),1,torch.device('cpu'));print('train',time.time()-t,flush=True)
t=time.time();rc=infer_raw(m,fold.X_cal,256,torch.device('cpu'));print('cal infer',time.time()-t,flush=True)
t=time.time();rt=infer_raw(m,fold.X_test,256,torch.device('cpu'));print('test infer',time.time()-t,flush=True)
t=time.time();cal=fit_calibrator(rc,fold.y_cal,CalibrationConfig(alpha=.025));print('fitcal',time.time()-t,cal,flush=True)
t=time.time();r=evaluate_variant(rt,fold.y_test,fold.unknown_test,cal,'complete',rc,fold.y_cal);print('eval',time.time()-t,{k:r[k] for k in ['known_macro_f1','unknown_auroc','open_set_macro_f1']},flush=True)
