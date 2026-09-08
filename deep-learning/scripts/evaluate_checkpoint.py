#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, torch
from pathlib import Path
from gateperceiver.models import GatePerceiver
from gateperceiver.training.engine import build_loader, evaluate_model, selection_score
from gateperceiver.training.report import build_evaluation_report
from gateperceiver.config import load_yaml
from gateperceiver.utils.io import write_json


def write_rows(rows, path):
    if not rows: return
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--checkpoint",required=True); ap.add_argument("--config"); ap.add_argument("--dataset-root"); ap.add_argument("--split",default="test",choices=["validation","test"]); ap.add_argument("--device",default="cuda"); ap.add_argument("--output",default="evaluation_test.json"); ap.add_argument("--per-sample-output"); args=ap.parse_args()
    ck=torch.load(args.checkpoint,map_location="cpu"); cfg=load_yaml(args.config) if args.config else ck["config"]
    if args.dataset_root: cfg["data"]["root"]=args.dataset_root
    dev=torch.device(args.device if args.device!="cuda" or torch.cuda.is_available() else "cpu"); ds,loader=build_loader(cfg,args.split,False); model=GatePerceiver(cfg); model.load_state_dict(ck["model"]); model.to(dev)
    m,rows=evaluate_model(model,loader,cfg,dev,return_rows=True); score=selection_score(m,cfg); report=build_evaluation_report(cfg,ds.dataset_meta,m,model,score,args.split); write_json(report,args.output)
    per=args.per_sample_output or str(Path(args.output).with_name("per_sample_metrics_test.csv" if args.split=="test" else "per_sample_metrics_validation.csv")); write_rows(rows,per)
    print(m); print("report:",args.output); print("per-sample:",per)
if __name__=="__main__": main()
