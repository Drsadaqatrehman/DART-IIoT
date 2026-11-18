import time, torch
from pathlib import Path
from real_benchmarks import load_unsw, prepare_explicit_fold
from dart_iiot import ModelConfig, TrainConfig, train_model

torch.set_num_threads(4)
t=time.time();df=load_unsw(Path('/mnt/data/UNSW_NB15_training-set.csv'),Path('/mnt/data/UNSW_NB15_testing-set.csv'),5000,2500);print('load',time.time()-t,flush=True)
t=time.time();fold=prepare_explicit_fold(df,'UNSW-NB15','DoS',16,4,False,32);print('prep',time.time()-t,fold.metadata,flush=True)
for e in [1,2,5]:
 t=time.time(); model,h=train_model(fold,ModelConfig(),TrainConfig(epochs=e,batch_size=256,patience=6),1,torch.device('cpu'));print('train',e,time.time()-t,h.seconds,flush=True)
