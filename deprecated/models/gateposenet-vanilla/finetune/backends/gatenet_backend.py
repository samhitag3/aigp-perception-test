"""GateNet fine-tuning backend (wraps gatenet.engine + gatenet.finetune)."""

from __future__ import annotations

from pathlib import Path

import torch

from gatenet.engine import build_datasets, fit, make_loaders
from gatenet.finetune import configure_finetune, param_groups
from gatenet.losses import DeepSupervisionLoss
from gatenet.model import build_gatenet
from gatenet.schedule import build_scheduler
from gatenet.utils import load_config, pick_device, set_seed

from .base import Backend


class GateNetBackend(Backend):
    name = "gatenet"

    def finetune(self, args) -> dict:
        if not args.config:
            raise SystemExit("gatenet backend requires --config (e.g. configs/gatenet_finetune.yaml)")
        if not args.pretrained:
            raise SystemExit("gatenet backend requires --pretrained <checkpoint>")

        cfg = load_config(args.config)
        ft = cfg.get("finetune", {})
        set_seed(cfg["train"].get("seed", 42))
        device = pick_device(args.device)
        epochs = args.epochs or cfg["train"]["epochs"]
        out_dir = Path(args.output or cfg.get("output", {}).get("dir", "runs/gatenet_finetune"))

        freeze_encoder = args.freeze_encoder or ft.get("freeze_encoder", False)
        reset_heads = args.reset_heads or ft.get("reset_heads", False)
        freeze_bn = args.freeze_bn_stats or ft.get("freeze_bn_stats", False)
        enc_lr_mult = ft.get("encoder_lr_mult", 0.1)

        train_ds, val_ds = build_datasets(cfg)
        train_loader, val_loader = make_loaders(cfg, train_ds, val_ds)

        model = build_gatenet(cfg["model"]).to(device)
        configure_finetune(model, args.pretrained, freeze_encoder=freeze_encoder,
                           reset_heads=reset_heads, freeze_bn_stats=freeze_bn,
                           map_location=device)

        criterion = DeepSupervisionLoss(
            scale_weights=cfg["loss"].get("scale_weights", (4, 2, 1, 1, 1)),
            bce_weight=cfg["loss"].get("bce_weight", 2.0))
        groups = param_groups(model, base_lr=cfg["train"]["lr"],
                              encoder_lr_mult=enc_lr_mult,
                              weight_decay=cfg["train"].get("weight_decay", 0.01))
        optimizer = torch.optim.AdamW(groups)
        scheduler = build_scheduler(optimizer, cfg.get("schedule"))

        print(f"[gatenet] fine-tune from {args.pretrained} | freeze_encoder={freeze_encoder} "
              f"reset_heads={reset_heads} enc_lr_mult={enc_lr_mult}")
        best = fit(cfg, model, train_loader, val_loader, criterion, optimizer,
                   scheduler, device, out_dir, epochs, tag="finetune")
        return {"backend": "gatenet", "best_iou": best, "out_dir": str(out_dir)}
