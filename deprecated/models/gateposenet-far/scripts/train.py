#!/usr/bin/env python3
"""Train GateNet from scratch.

Usage:
    python scripts/train.py --config configs/gatenet_a2rl.yaml
    python scripts/train.py --config configs/gatenet_a2rl.yaml --epochs 5 --device cpu

Data is consumed from the standard images/ + masks/ contract (see data/README.md).
To adapt a pretrained model to a new gate type / venue, use scripts/finetune.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatenet.engine import build_datasets, fit, make_loaders, make_optimizer
from gatenet.losses import DeepSupervisionLoss
from gatenet.model import build_gatenet
from gatenet.schedule import build_scheduler
from gatenet.utils import load_checkpoint, load_config, pick_device, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", default=None, help="resume the SAME run (weights+optim+epoch)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["train"].get("seed", 42))
    device = pick_device(args.device)
    epochs = args.epochs or cfg["train"]["epochs"]
    out_dir = Path(cfg.get("output", {}).get("dir", "runs/gatenet"))

    train_ds, val_ds = build_datasets(cfg)
    train_loader, val_loader = make_loaders(cfg, train_ds, val_ds)

    model = build_gatenet(cfg["model"]).to(device)
    print(f"GateNet parameters: {model.num_parameters()/1e6:.3f} M  | device: {device}")
    criterion = DeepSupervisionLoss(
        scale_weights=cfg["loss"].get("scale_weights", (4, 2, 1, 1, 1)),
        bce_weight=cfg["loss"].get("bce_weight", 2.0))
    optimizer = make_optimizer(cfg, model)
    scheduler = build_scheduler(optimizer, cfg.get("schedule"))

    start_epoch, best_iou = 0, 0.0
    if args.resume:
        ck = load_checkpoint(args.resume, model, optimizer, scheduler, map_location=device)
        start_epoch = ck.get("epoch", 0) + 1
        best_iou = ck.get("best_metric", 0.0)
        print(f"resumed from {args.resume} @ epoch {start_epoch}")

    best = fit(cfg, model, train_loader, val_loader, criterion, optimizer,
               scheduler, device, out_dir, epochs, start_epoch, best_iou, tag="train")
    print(f"done. best val IoU = {best:.4f}")


if __name__ == "__main__":
    main()
