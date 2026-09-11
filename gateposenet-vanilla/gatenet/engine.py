"""Shared training engine used by both scripts/train.py and scripts/finetune.py.

Keeps the data-building, optimizer-building, validation and the train/eval loop
in one place so training and fine-tuning behave identically.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .augment import build_augment
from .dataset import GateSegDataset, MergedGateDataset, split_dataset
from .metrics import seg_metrics
from .utils import AverageMeter, save_checkpoint


def build_datasets(cfg):
    """Build (train_ds, val_ds) from cfg['data']. Supports 'single'/'merged'."""
    d = cfg["data"]
    size = (d["height"], d["width"])
    train_aug = build_augment(cfg.get("augment"), train=True)
    val_aug = build_augment(cfg.get("augment"), train=False)
    suffix = d.get("mask_suffix", "")

    if d.get("mode", "single") == "merged":
        synth = GateSegDataset(d["synth_root"], size, augment=train_aug, mask_suffix=suffix)
        real = GateSegDataset(d["real_root"], size, augment=train_aug, mask_suffix=suffix)
        train_ds = MergedGateDataset(synth, real,
                                     ratio=tuple(d.get("ratio", (3500, 500))),
                                     length=len(synth) + len(real))
        real_val = GateSegDataset(d["real_root"], size, augment=val_aug, mask_suffix=suffix)
        _, val_ds = split_dataset(real_val, d.get("val_frac", 0.1))
        return train_ds, val_ds

    full = GateSegDataset(d["root"], size, augment=train_aug, mask_suffix=suffix)
    train_ds, val_sub = split_dataset(full, d.get("val_frac", 0.1))
    val_full = GateSegDataset(d["root"], size, augment=val_aug, mask_suffix=suffix)
    val_ds = Subset(val_full, val_sub.indices)
    return train_ds, val_ds


def make_loaders(cfg, train_ds, val_ds):
    t = cfg["train"]
    nw = t.get("num_workers", 4)
    train_loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=t["batch_size"], shuffle=False,
                            num_workers=nw, pin_memory=True)
    return train_loader, val_loader


def make_optimizer(cfg, params):
    """Build an optimizer from cfg['train']. `params` may be a model or param groups."""
    t = cfg["train"]
    if hasattr(params, "parameters"):
        params = params.parameters()
    name = t.get("optimizer", "adamw").lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=t["lr"], weight_decay=t.get("weight_decay", 0.01))
    if name == "adam":
        return torch.optim.Adam(params, lr=t["lr"])
    if name == "sgd":
        return torch.optim.SGD(params, lr=t["lr"], momentum=0.9,
                               weight_decay=t.get("weight_decay", 0.0))
    raise ValueError(f"unknown optimizer {name}")


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_m, iou_m, dice_m = AverageMeter(), AverageMeter(), AverageMeter()
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        out = model(img)
        loss, _ = criterion(out, mask)
        m = seg_metrics(out[0], mask)
        loss_m.update(loss.item(), img.size(0))
        iou_m.update(m["iou"], img.size(0))
        dice_m.update(m["dice"], img.size(0))
    return {"loss": loss_m.avg, "iou": iou_m.avg, "dice": dice_m.avg}


def fit(cfg, model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, out_dir, epochs, start_epoch=0, best_iou=0.0, tag="train"):
    """Run the train/validate/checkpoint loop. Returns best val IoU."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_amp = cfg["train"].get("amp", True) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    log_int = cfg["train"].get("log_interval", 20)

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(out_dir / "tb")
    except Exception:
        pass

    for epoch in range(start_epoch, epochs):
        model.train()
        loss_m = AverageMeter()
        pbar = tqdm(train_loader, desc=f"[{tag}] epoch {epoch+1}/{epochs}")
        for it, (img, mask) in enumerate(pbar):
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            amp_ctx = torch.cuda.amp.autocast() if use_amp else contextlib.nullcontext()
            with amp_ctx:
                out = model(img)
                loss, _ = criterion(out, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_m.update(loss.item(), img.size(0))
            if it % log_int == 0:
                pbar.set_postfix(loss=f"{loss_m.avg:.4f}",
                                 lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        if scheduler is not None:
            scheduler.step()

        val = validate(model, val_loader, criterion, device)
        print(f"  [val] loss={val['loss']:.4f}  IoU={val['iou']:.4f}  Dice={val['dice']:.4f}")
        if writer:
            writer.add_scalar(f"{tag}/loss", loss_m.avg, epoch)
            writer.add_scalar("val/loss", val["loss"], epoch)
            writer.add_scalar("val/iou", val["iou"], epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        save_checkpoint(out_dir / "last.pt", model, optimizer, scheduler,
                        epoch=epoch, best_metric=best_iou, extra={"config": cfg})
        if val["iou"] > best_iou:
            best_iou = val["iou"]
            save_checkpoint(out_dir / "best.pt", model, optimizer, scheduler,
                            epoch=epoch, best_metric=best_iou, extra={"config": cfg})
            print(f"  * new best IoU={best_iou:.4f} -> {out_dir/'best.pt'}")

    return best_iou
