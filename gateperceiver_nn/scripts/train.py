#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from gateperceiver.config import load_yaml, deep_merge, save_yaml
from gateperceiver.training import train_experiment


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--overrides")
    ap.add_argument("--dataset-root")
    ap.add_argument("--run-dir",required=True)
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--epochs",type=int)
    args=ap.parse_args()
    cfg=load_yaml(args.config)
    if args.overrides: cfg=deep_merge(cfg,load_yaml(args.overrides))
    if args.dataset_root: cfg.setdefault("data",{})["root"]=args.dataset_root
    if args.epochs is not None: cfg.setdefault("training",{})["epochs"]=args.epochs
    Path(args.run_dir).mkdir(parents=True,exist_ok=True); save_yaml(cfg,Path(args.run_dir)/"resolved_config.yaml")
    score,metrics=train_experiment(cfg,args.run_dir,args.device)
    print(f"best composite score: {score:.6f}")
    print(metrics)
if __name__=="__main__": main()
