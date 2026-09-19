#!/usr/bin/env python3
"""Optuna tuning for unchanged vanilla GatePoseNetSingle.

Only optimizer/loss/dropout training hyperparameters are tuned. Model architecture
parameters are intentionally not tuned so this remains the vanilla architecture.
The training subset is supplied by a fixed sequence manifest (normally the same
50% manifest used by the baseline run).
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import optuna
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatenet.schedule import build_scheduler
from gatenet.utils import load_config, pick_device, set_seed
from gateposenet.engine import build_datasets, fit, make_loaders, make_optimizer
from gateposenet.losses import GatePoseLoss
from gateposenet.model_single import build_gateposenet_single


def set_manifest(cfg: dict, kind: str, value: str) -> None:
    key = {"train": "train_manifest", "validation": "val_manifest", "test": "test_manifest"}[kind]
    cfg["data"][key] = value
    for src in cfg["data"].get("sources", []):
        src[key] = value


def scaled_schedule(epochs: int, gamma: float) -> dict:
    # Preserve the original three-decay shape while scaling it to short trials.
    pts = sorted(set(max(1, min(epochs - 1, round(frac * epochs)))
                     for frac in (1/3, 2/3, 0.87)))
    return {"milestones": pts, "gamma": gamma}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-manifest", default="splits/vanilla_seed42/train_tune_sequences.txt")
    ap.add_argument("--val-manifest", default="splits/validation_sequences.txt")
    ap.add_argument("--run-dir", required=True,
                    help="trial checkpoints: <run-dir>/trial_XXXX")
    ap.add_argument("--output-dir", required=True,
                    help="study DB, CSV, best_config.yaml, best_params.json")
    ap.add_argument("--device", default=None)
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--epochs-per-trial", type=int, default=None)
    ap.add_argument("--study-name", default=None)
    args = ap.parse_args()

    base = load_config(args.config)
    set_manifest(base, "train", args.train_manifest)
    set_manifest(base, "validation", args.val_manifest)

    tune_cfg = base.get("tuning", {})
    n_trials = int(args.trials or tune_cfg.get("trials", 20))
    epochs = int(args.epochs_per_trial or tune_cfg.get("epochs_per_trial", 15))
    seed = int(tune_cfg.get("seed", base.get("train", {}).get("seed", 42)))
    device = pick_device(args.device)

    run_root = Path(args.run_dir)
    out_root = Path(args.output_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    study_name = args.study_name or f"{base.get('name', 'gatepose')}_optuna"
    storage = f"sqlite:///{(out_root / 'study.db').resolve()}"
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=str(tune_cfg.get("direction", "minimize")),
        sampler=sampler,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        cfg = copy.deepcopy(base)
        cfg["train"]["seed"] = seed
        # Optimizer/search settings only; model architecture stays untouched.
        cfg["train"]["lr"] = trial.suggest_float("lr", 2e-4, 3e-3, log=True)
        cfg["train"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-5, 5e-2, log=True)
        cfg["train"]["grad_clip"] = trial.suggest_float("grad_clip", 0.5, 2.0)
        cfg["data"]["ego_dropout"] = trial.suggest_float("ego_dropout", 0.0, 0.40)

        cfg["loss"]["w_seg"] = trial.suggest_float("w_seg", 0.5, 2.0)
        cfg["loss"]["w_corners"] = trial.suggest_float("w_corners", 2.0, 8.0)
        cfg["loss"]["w_position"] = trial.suggest_float("w_position", 1.0, 5.0)
        cfg["loss"]["w_depth"] = trial.suggest_float("w_depth", 0.5, 4.0)
        cfg["loss"]["w_rot"] = trial.suggest_float("w_rot", 0.5, 4.0)
        cfg["loss"]["w_vis"] = trial.suggest_float("w_vis", 0.5, 2.0)

        cfg["schedule"] = scaled_schedule(
            epochs, float(base.get("schedule", {}).get("gamma", 0.31622776601)))

        trial_dir = run_root / f"trial_{trial.number:04d}"
        cfg.setdefault("output", {})["dir"] = str(trial_dir)
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        set_seed(cfg["train"]["seed"])
        train_ds, val_ds = build_datasets(cfg)
        train_loader, val_loader = make_loaders(cfg, train_ds, val_ds)
        model = build_gateposenet_single(cfg["model"]).to(device)
        criterion = GatePoseLoss(**cfg.get("loss", {})).to(device)
        optimizer = make_optimizer(cfg, model)
        scheduler = build_scheduler(optimizer, cfg.get("schedule"))

        best = fit(
            cfg, model, train_loader, val_loader, criterion, optimizer,
            scheduler, device, trial_dir, epochs, 0, float("inf"),
            tag=f"optuna-{trial.number}")
        trial.set_user_attr("best_checkpoint", str(trial_dir / "best.pt"))
        return float(best)

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    best_cfg = copy.deepcopy(base)
    bp = study.best_params
    best_cfg["train"]["lr"] = bp["lr"]
    best_cfg["train"]["weight_decay"] = bp["weight_decay"]
    best_cfg["train"]["grad_clip"] = bp["grad_clip"]
    best_cfg["data"]["ego_dropout"] = bp["ego_dropout"]
    for k in ("w_seg", "w_corners", "w_position", "w_depth", "w_rot", "w_vis"):
        best_cfg["loss"][k] = bp[k]
    # Final training returns to the full 60-epoch/default schedule and official
    # train split; the final command may override these explicitly too.
    set_manifest(best_cfg, "train", "splits/train_sequences.txt")
    best_cfg["train"]["seed"] = int(base.get("train", {}).get("seed", 42))
    best_cfg["train"]["epochs"] = int(base.get("train", {}).get("epochs", 60))
    best_cfg["schedule"] = copy.deepcopy(base.get("schedule"))

    (out_root / "best_config.yaml").write_text(
        yaml.safe_dump(best_cfg, sort_keys=False), encoding="utf-8")
    (out_root / "best_params.json").write_text(json.dumps({
        "study_name": study.study_name,
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "objective": tune_cfg.get("objective", "pos_err_rel"),
        "params": study.best_params,
    }, indent=2) + "\n", encoding="utf-8")

    rows = []
    for t in study.trials:
        row = {"trial": t.number, "state": t.state.name, "value": t.value}
        row.update(t.params)
        rows.append(row)
    fieldnames = sorted({k for r in rows for k in r}) if rows else ["trial", "state", "value"]
    with (out_root / "trials.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"best trial={study.best_trial.number} objective={study.best_value:.6f}")
    print(f"best config -> {out_root / 'best_config.yaml'}")
    print(f"best params -> {out_root / 'best_params.json'}")
    print(f"study db -> {out_root / 'study.db'}")


if __name__ == "__main__":
    main()
