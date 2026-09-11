#!/usr/bin/env python3
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import optuna
from gateperceiver.config import load_yaml, save_yaml
from gateperceiver.training import train_experiment


def put(d,path,value):
    cur=d
    parts=path.split(".")
    for p in parts[:-1]: cur=cur.setdefault(p,{})
    cur[parts[-1]]=value


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--dataset-root", action="append", dest="dataset_roots",
                    help="Canonical dataset root. Repeat this option to combine datasets.")
    ap.add_argument("--run-dir",required=True)
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--n-trials",type=int,default=None)
    args=ap.parse_args()
    tc=load_yaml(args.config); base=load_yaml(tc["base_config"])
    if args.dataset_roots:
        base.setdefault("data",{})["roots"]=args.dataset_roots
        base["data"].pop("root",None)
    out=Path(args.run_dir); out.mkdir(parents=True,exist_ok=True)
    seed=int(base.get("seed",42))
    sampler=optuna.samplers.TPESampler(seed=seed)
    study=optuna.create_study(
        direction="maximize", study_name=tc.get("study_name","gateperceiver"),
        storage=tc.get("storage"), load_if_exists=True, sampler=sampler,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=int(tc.get("n_startup_trials",5)),
            n_warmup_steps=int(tc.get("n_warmup_epochs",2)),
        ),
    )
    space=tc["search_space"]
    def objective(trial):
        cfg=deepcopy(base); cfg["training"]["epochs"]=int(tc.get("trial_epochs",8))
        for path,s in space.items():
            typ=s["type"]
            if typ=="float": val=trial.suggest_float(path,float(s["low"]),float(s["high"]),log=bool(s.get("log",False)))
            elif typ=="int": val=trial.suggest_int(path,int(s["low"]),int(s["high"]),step=int(s.get("step",1)))
            elif typ=="categorical": val=trial.suggest_categorical(path,s["choices"])
            else: raise ValueError(typ)
            put(cfg,path,val)
        score,_=train_experiment(cfg,out/f"trial_{trial.number:04d}",args.device,trial=trial)
        return score
    n=args.n_trials if args.n_trials is not None else int(tc.get("n_trials",20))
    study.optimize(objective,n_trials=n)
    overrides={}
    for path,val in study.best_params.items(): put(overrides,path,val)
    save_yaml(overrides,out/"best_overrides.yaml")
    save_yaml({"seed":seed,"best_value":float(study.best_value),"best_trial":int(study.best_trial.number),"best_params":study.best_params},out/"study_summary.yaml")
    print("best value",study.best_value); print("best params",study.best_params); print("saved",out/"best_overrides.yaml")
if __name__=="__main__": main()
