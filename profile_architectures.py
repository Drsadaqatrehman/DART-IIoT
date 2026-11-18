import time, numpy as np, pandas as pd, torch
from dart_iiot import DARTIIoT, ModelConfig
from real_benchmarks import MLPClassifier, GRUClassifier, CNNGRUClassifier

torch.set_num_threads(1)
def profile(model, x, repeats=300):
    model.eval()
    with torch.no_grad():
        for _ in range(30): model(x)
        vals=[]
        for _ in range(repeats):
            t=time.perf_counter(); model(x); vals.append((time.perf_counter()-t)*1000)
    p=sum(z.numel() for z in model.parameters())
    return p, p*4/1024, p/1024, float(np.median(vals)), float(np.quantile(vals,.95))
rows=[]
x=torch.randn(1,16,32)
for dataset,k in [('UNSW-NB15',9),('N-BaIoT',2)]:
    models=[('DART-IIoT',DARTIIoT(32,k,ModelConfig())),('MLP-MSP',MLPClassifier(16,32,k)),('GRU-MSP',GRUClassifier(32,k)),('CNN-GRU-MSP',CNNGRUClassifier(32,k))]
    for name,m in models:
        p,fp,int8,med,p95=profile(m,x)
        rows.append(dict(dataset=dataset,method=name,parameters=p,fp32_kib=fp,estimated_int8_kib=int8,median_ms=med,p95_ms=p95))
df=pd.DataFrame(rows); df.to_csv('/mnt/data/DART_IIoT_real_experiments/aggregate/architecture_profile.csv',index=False); print(df.to_string(index=False))
