from __future__ import annotations
import numpy as np, torch

@torch.no_grad()
def benchmark_step(model, device, height=192, width=320, warmup=40, iters=120):
    model.eval()
    x=torch.zeros(1,3,height,width,device=device)
    ego=torch.zeros(1,7,device=device); ego[:,6]=1/60
    h=None
    for _ in range(warmup):
        _,h=model.step(x,ego,h)
    if device.type=='cuda': torch.cuda.synchronize()
    ms=[]
    for _ in range(iters):
        if device.type=='cuda':
            a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True); a.record()
            _,h=model.step(x,ego,h); b.record(); torch.cuda.synchronize(); ms.append(a.elapsed_time(b))
        else:
            import time; t=time.perf_counter(); _,h=model.step(x,ego,h); ms.append((time.perf_counter()-t)*1000)
    x=np.asarray(ms,dtype=float)
    return {'mean_ms':float(x.mean()),'median_ms':float(np.median(x)),'p95_ms':float(np.percentile(x,95)),'p99_ms':float(np.percentile(x,99)),'mean_hz':float(1000/x.mean())}
