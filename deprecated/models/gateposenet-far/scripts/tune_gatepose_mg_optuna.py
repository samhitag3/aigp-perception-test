#!/usr/bin/env python3
"""Optuna tuning for GatePoseNet-MG using the fixed canonical 50% subset."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import optuna
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatenet.utils import load_config, pick_device, set_seed
from gateposenet.losses_mg import GatePoseMGLoss
from gateposenet.mg_engine_contract import (fit_mg, make_loaders, make_optimizer,
                                            make_scheduler, set_manifest)
from gateposenet.model import build_gateposenet_mg


def scaled_schedule(epochs: int, gamma: float) -> dict:
    pts = sorted(set(max(1, min(epochs - 1, round(frac * epochs)))
                     for frac in (1/3, 2/3, 0.87)))
    return {"milestones": pts, "gamma": gamma}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-manifest", default="splits/vanilla_seed42/train_tune_sequences.txt")
    ap.add_argument("--val-manifest", default="splits/validation_sequences.txt")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--epochs-per-trial", type=int, default=None)
    ap.add_argument("--study-name", default=None)
    args = ap.parse_args()

    base = load_config(args.config)
    set_manifest(base, "train", args.train_manifest)
    set_manifest(base, "validation", args.val_manifest)
    tune = base.get("tuning", {})
    trials = int(args.trials or tune.get("trials", 20))
    epochs = int(args.epochs_per_trial or tune.get("epochs_per_trial", 15))
    seed = int(tune.get("seed", base["train"].get("seed", 42)))
    device = pick_device(args.device)
    run_root, out_root = Path(args.run_dir), Path(args.output_dir)
    run_root.mkdir(parents=True, exist_ok=True); out_root.mkdir(parents=True, exist_ok=True)
    study_name = args.study_name or f"{base.get('name','gatepose_mg')}_optuna"
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{(out_root/'study.db').resolve()}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        load_if_exists=True,
    )

    def objective(trial):
        cfg = copy.deepcopy(base)
        cfg["train"]["seed"] = seed
        cfg["train"]["lr"] = trial.suggest_float("lr", 1.5e-4, 2e-3, log=True)
        cfg["train"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-5, 5e-2, log=True)
        cfg["train"]["grad_clip"] = trial.suggest_float("grad_clip", 0.5, 2.0)
        cfg["data"]["ego_dropout"] = trial.suggest_float("ego_dropout", 0.0, 0.40)
        for key, lo, hi in [
            ("w_presence", 1.0, 4.0), ("w_mask", 1.0, 4.0),
            ("w_corners", 2.0, 8.0), ("w_position", 1.0, 5.0),
            ("w_depth", 0.5, 4.0), ("w_rot", 0.5, 4.0),
            ("w_target", 0.5, 3.0)]:
            cfg["loss"][key] = trial.suggest_float(key, lo, hi)
        cfg["schedule"] = scaled_schedule(epochs, float(base.get("schedule", {}).get("gamma", 0.31622776601)))
        trial_dir = run_root / f"trial_{trial.number:04d}"
        cfg.setdefault("output", {})["dir"] = str(trial_dir)
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir/"config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        set_seed(seed)
        _, _, tl, vl = make_loaders(cfg)
        model = build_gateposenet_mg(cfg["model"]).to(device)
        criterion = GatePoseMGLoss(**cfg.get("loss", {})).to(device)
        optimizer = make_optimizer(cfg, model); scheduler = make_scheduler(cfg, optimizer)
        best = fit_mg(cfg, model, tl, vl, criterion, optimizer, scheduler,
                      device, trial_dir, epochs, tag=f"mg-optuna-{trial.number}")
        trial.set_user_attr("best_checkpoint", str(trial_dir/"best.pt"))
        return float(best)

    study.optimize(objective, n_trials=trials, gc_after_trial=True)
    bp = study.best_params
    best_cfg = copy.deepcopy(base)
    for k, v in bp.items():
        if k in {"lr", "weight_decay", "grad_clip"}:
            best_cfg["train"][k] = v
        elif k == "ego_dropout":
            best_cfg["data"][k] = v
        else:
            best_cfg["loss"][k] = v
    set_manifest(best_cfg, "train", "splits/train_sequences.txt")
    best_cfg["train"]["seed"] = int(base["train"].get("seed", 42))
    best_cfg["train"]["epochs"] = int(base["train"].get("epochs", 60))
    best_cfg["schedule"] = copy.deepcopy(base.get("schedule"))
    (out_root/"best_config.yaml").write_text(yaml.safe_dump(best_cfg, sort_keys=False), encoding="utf-8")
    (out_root/"best_params.json").write_text(json.dumps({
        "study_name": study.study_name, "best_trial": study.best_trial.number,
        "best_value": study.best_value, "objective": base.get("tuning", {}).get("objective", "target_pos_err_m"),
        "params": study.best_params}, indent=2)+"\n", encoding="utf-8")
    rows=[]
    for t in study.trials:
        r={"trial":t.number,"state":t.state.name,"value":t.value}; r.update(t.params); rows.append(r)
    fields=sorted({k for r in rows for k in r}) if rows else ["trial","state","value"]
    with (out_root/"trials.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    objective_name = base.get("tuning", {}).get("objective", "target_pos_err_m")
    print(f"best trial={study.best_trial.number} {objective_name}={study.best_value:.6f}")
    print(f"best config -> {out_root/'best_config.yaml'}")


if __name__ == "__main__":
    main()
