from __future__ import annotations
import argparse
from pathlib import Path
import optuna
import torch

from gateperception.models import build_keypoint_model
from gateperception.training.builders import build_keypoint_loaders, keypoint_step, keypoint_val
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
                "learning_rate": trial.suggest_float("lr", 1e-5, 8e-4, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True),
            },
            "model": {"crop_padding": trial.suggest_float("crop_padding", 0.20, 0.60)},
            "loss": {
                "coord_weight": trial.suggest_float("coord_weight", 2.0, 8.0),
                "visibility_weight": trial.suggest_float("visibility_weight", 0.5, 2.0),
            },
        }
        cfg = deep_update(base, patch)
        seed_everything(int(cfg.get("seed", 42)) + trial.number)
        _, _, tr, va = build_keypoint_loaders(cfg)
        model = build_keypoint_model(cfg).to(device)
        if args.init_checkpoint:
            ck = torch.load(args.init_checkpoint, map_location="cpu")
            model.load_state_dict(ck["model"], strict=True)
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
        run = study_dir / f"trial_{trial.number:04d}"
        result = train_loop(model, tr, va, opt, int(cfg["training"]["epochs"]), device, run, cfg, keypoint_step, keypoint_val, amp=bool(cfg["training"].get("amp", True)))
        return float(result["best_metric"])

    study = optuna.create_study(study_name=base.get("optuna", {}).get("study_name", "temporal_gate_keypoints"), storage=db, load_if_exists=True, direction="minimize")
    study.optimize(objective, n_trials=args.n_trials)
    p = study.best_params
    best = deep_update(base, {
        "training": {"learning_rate": p["lr"], "weight_decay": p["weight_decay"]},
        "model": {"crop_padding": p["crop_padding"]},
        "loss": {"coord_weight": p["coord_weight"], "visibility_weight": p["visibility_weight"]},
    })
    save_yaml(best, study_dir / "best_config.yaml")
    print("best_value=", study.best_value)
    print("best_params=", study.best_params)

if __name__ == "__main__":
    main()
