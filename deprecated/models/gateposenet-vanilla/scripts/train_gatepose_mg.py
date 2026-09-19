#!/usr/bin/env python3
"""Train GatePoseNet-MG (multi-gate query decoder).

    uv run python scripts/train_gatepose_mg.py --config configs/gateposenet_mg.yaml

Validation tracks: TARGET-gate position error (the racing quantity), mean
per-gate mask IoU, presence accuracy — best.pt selected on target pos err.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatenet.schedule import build_scheduler
from gatenet.utils import (AverageMeter, load_config, pick_device,
                           save_checkpoint, set_seed)
from gateposenet.dataset_mg import GateSequenceDatasetMG
from gateposenet.losses_mg import GatePoseMGLoss
from gateposenet.model import build_gateposenet_mg


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    lm = AverageMeter()
    tp_err, iou_all, pres_ok = [], [], []
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        pred = model(batch["image"], batch["ego"])
        loss, _ = criterion(pred, batch)
        lm.update(float(loss), n=batch["image"].shape[0])

        # target-gate position error via the model's own target head
        q_sel = pred["target_logit"].argmax(dim=2)            # (B,T)
        pos_q = pred["position"].gather(
            2, q_sel[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
        t_idx = batch["g_target"].argmax(dim=2)               # (B,T)
        pos_gt = batch["g_position"].gather(
            2, t_idx[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
        sel = ((batch["g_target"].max(dim=2).values > 0.5)
               & (batch["frame_ok"] > 0.5)
               & (batch["g_visible"].gather(
                   2, t_idx[..., None]).squeeze(-1) > 0.5))
        if sel.any():
            tp_err.append((pos_q - pos_gt).norm(dim=-1)[sel].cpu())

        # presence accuracy + matched-mask IoU (greedy vs GT for speed)
        prob = torch.sigmoid(pred["presence_logit"])
        n_pred = (prob > 0.5).sum(dim=2).float()
        n_gt = batch["g_valid"].sum(dim=2)
        pres_ok.append(((n_pred == n_gt)[batch["frame_ok"] > 0.5])
                       .float().cpu())
        pm = torch.sigmoid(pred["mask_logit"]) > 0.5
        for b in range(pm.shape[0]):
            for t in range(pm.shape[1]):
                if batch["frame_ok"][b, t] < 0.5:
                    continue
                for g in range(batch["g_valid"].shape[2]):
                    if batch["g_valid"][b, t, g] < 0.5 or \
                            batch["g_mask"][b, t, g].sum() < 4:
                        continue
                    gm = batch["g_mask"][b, t, g] > 0.5
                    ious = ((pm[b, t] & gm).sum(dim=(-1, -2)).float()
                            / ((pm[b, t] | gm).sum(dim=(-1, -2))
                               .float() + 1e-6))
                    iou_all.append(float(ious.max()))
    out = {"val_loss": lm.avg}
    out["target_pos_err_m"] = (float(torch.cat(tp_err).median())
                               if tp_err else float("inf"))
    out["mask_iou"] = (sum(iou_all) / len(iou_all)) if iou_all else 0.0
    out["presence_acc"] = (float(torch.cat(pres_ok).mean())
                           if pres_ok else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gateposenet_mg.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["train"].get("seed", 42))
    device = pick_device(args.device)
    d = cfg["data"]
    size = (int(d.get("height", 192)), int(d.get("width", 320)))
    common = dict(window=int(d.get("window", 8)), size=size,
                  pose_sup_max_m=float(d.get("pose_sup_max_m", 25.0)))
    train_ds = GateSequenceDatasetMG(d["train_roots"],
                                     stride=int(d.get("stride", 4)),
                                     ego_dropout=float(
                                         d.get("ego_dropout", 0.25)),
                                     **common)
    val_ds = GateSequenceDatasetMG(d["val_roots"],
                                   stride=int(d.get("val_stride", 8)),
                                   **common)
    print(f"windows: train={len(train_ds)} val={len(val_ds)}")
    kw = dict(num_workers=int(cfg["train"].get("num_workers", 8)),
              pin_memory=True, persistent_workers=True)
    tl = DataLoader(train_ds, batch_size=int(cfg["train"]["batch_size"]),
                    shuffle=True, drop_last=True, **kw)
    vl = DataLoader(val_ds, batch_size=int(cfg["train"]["batch_size"]), **kw)

    model = build_gateposenet_mg(cfg["model"]).to(device)
    print(f"GatePoseNet-MG parameters: {model.num_parameters()/1e6:.2f} M")
    criterion = GatePoseMGLoss(**cfg.get("loss", {})).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(cfg["train"].get("lr", 1e-3)),
                                  weight_decay=float(
                                      cfg["train"].get("weight_decay", 0.01)))
    scheduler = build_scheduler(optimizer, cfg.get("schedule"))
    out_dir = Path(cfg.get("output", {}).get("dir", "runs/gateposenet_mg"))
    out_dir.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(str(out_dir / "tb"))

    epochs = args.epochs or int(cfg["train"]["epochs"])
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best = float("inf")
    step = 0
    for epoch in range(epochs):
        model.train()
        meter = AverageMeter()
        t0 = time.time()
        for batch in tqdm(tl, desc=f"[mg] epoch {epoch+1}/{epochs}",
                          leave=False):
            batch = {k: v.to(device, non_blocking=True)
                     for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                pred = model(batch["image"], batch["ego"])
                loss, logs = criterion(pred, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            meter.update(float(loss), n=batch["image"].shape[0])
            step += 1
            if step % 20 == 0:
                for k, v in logs.items():
                    tb.add_scalar(f"train/{k}", v, step)
        metrics = validate(model, vl, criterion, device)
        for k, v in metrics.items():
            tb.add_scalar(f"val/{k}", v, epoch)
        if scheduler is not None:
            scheduler.step()
        print(f"[mg] epoch {epoch+1}/{epochs} train_loss={meter.avg:.4f} "
              f"val_loss={metrics['val_loss']:.4f} "
              f"target_pos={metrics['target_pos_err_m']:.3f}m "
              f"mask_iou={metrics['mask_iou']:.3f} "
              f"presence_acc={metrics['presence_acc']:.3f} "
              f"({time.time()-t0:.0f}s)")
        sel = metrics["target_pos_err_m"]
        extra = {"config": cfg, "val_metrics": metrics}
        if sel < best:
            best = sel
            save_checkpoint(out_dir / "best.pt", model, optimizer, scheduler,
                            epoch=epoch, best_metric=best, extra=extra)
        save_checkpoint(out_dir / "last.pt", model, optimizer, scheduler,
                        epoch=epoch, best_metric=best, extra=extra)
    tb.close()
    print(f"done. best target_pos_err={best:.3f} m -> {out_dir}/best.pt")


if __name__ == "__main__":
    main()
