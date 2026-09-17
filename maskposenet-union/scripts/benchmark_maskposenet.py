#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from gatenet.utils import load_config,load_checkpoint,pick_device
from gateposenet.model import build_gateposenet_mg
from gateposenet.latency import benchmark_step

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--checkpoint'); ap.add_argument('--device',default='cuda'); ap.add_argument('--warmup',type=int,default=50); ap.add_argument('--iters',type=int,default=300); ap.add_argument('--out',default=None); a=ap.parse_args()
 c=load_config(a.config); d=pick_device(a.device); m=build_gateposenet_mg(c['model']).to(d).eval()
 if a.checkpoint: load_checkpoint(a.checkpoint,m,map_location=d)
 r=benchmark_step(m,d,int(c['data'].get('height',192)),int(c['data'].get('width',320)),a.warmup,a.iters); r['target_90hz_budget_ms']=1000/90; r['meets_90hz_on_this_device']=r['p95_ms']<=1000/90
 if d.type=='cuda':
  try:r['gpu_util_percent_sample']=int(__import__('torch').cuda.utilization(d))
  except Exception:r['gpu_util_percent_sample']=None
  import torch; r['max_memory_allocated_mb']=torch.cuda.max_memory_allocated()/2**20
 print(json.dumps(r,indent=2))
 if a.out: Path(a.out).write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':main()
