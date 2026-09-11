from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import optuna
import torch

from gateperception.models import build_segmentation_model
from gateperception.training.builders import build_seg_loaders, seg_step, seg_val
from gateperception.training.engine import train_loop
from gateperception.utils.config import load_yaml, save_yaml, deep_update
from gateperception.utils.seed import seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--study-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-trials", type=int, default=25)
    ap.add_argument("--init-checkpoint", default=None)
    args = ap.parse_args()
    base = load_yaml(args.config)
    study_dir = Path(args.study_dir); study_dir.mkdir(parents=True, exist_ok=True)
    db = f"sqlite:///{(study_dir / 'study.db').resolve()}"
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    def objective(trial: optuna.Trial):
        patch = {
            "training": {
                "learning_rate": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True),
            },
            "loss": {
                "mask_dice_weight": trial.suggest_float("mask_dice_weight", 1.5, 5.0),
                "track_weight": trial.suggest_float("track_weight", 0.05, 0.6),
                "object_pos_weight": trial.suggest_float("object_pos_weight", 1.0, 4.0),
            },
        }
        cfg = deep_update(base, patch)
        # Keep training stochasticity fixed across trials; only hyperparameters vary.
        seed_everything(int(cfg.get("seed", 42)))
        _, _, tr, va = build_seg_loaders(cfg)
        model = build_segmentation_model(cfg).to(device)
        if args.init_checkpoint:
            ck = torch.load(args.init_checkpoint, map_location="cpu")
            model.load_state_dict(ck["model"], strict=True)
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
        run = study_dir / f"trial_{trial.number:04d}"
        result = train_loop(model, tr, va, opt, int(cfg["training"]["epochs"]), device, run, cfg, seg_step, seg_val, amp=bool(cfg["training"].get("amp", True)))
        return float(result["best_metric"])

    sampler = optuna.samplers.TPESampler(seed=int(base.get("seed", 42)))
    study = optuna.create_study(study_name=base.get("optuna", {}).get("study_name", "temporal_gate_seg"), storage=db, load_if_exists=True, direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=args.n_trials)
    best = deep_update(base, {
        "training": {"learning_rate": study.best_params["lr"], "weight_decay": study.best_params["weight_decay"]},
        "loss": {
            "mask_dice_weight": study.best_params["mask_dice_weight"],
            "track_weight": study.best_params["track_weight"],
            "object_pos_weight": study.best_params["object_pos_weight"],
        },
    })
    save_yaml(best, study_dir / "best_config.yaml")
    print("best_value=", study.best_value)
    print("best_params=", study.best_params)

if __name__ == "__main__":
    main()
