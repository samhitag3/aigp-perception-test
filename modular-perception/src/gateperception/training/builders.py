from __future__ import annotations
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import DataLoader

from gateperception.data.canonical import TemporalSegDataset, GateKeypointDataset, collate_seg, collate_keypoints
from gateperception.models import build_segmentation_model, build_keypoint_model
from gateperception.training.seg_losses import segmentation_loss
from gateperception.training.keypoint_losses import keypoint_loss


def build_seg_loaders(cfg: dict[str, Any]):
    d = cfg["dataset"]
    common = dict(
        root=d["root"],
        window_size=int(cfg["model"].get("window_size", 4)),
        width=int(d.get("width", 640)),
        height=int(d.get("height", 360)),
        augmentation=cfg.get("augmentation"),
        seed=int(cfg.get("seed", 42)),
    )
    train_ds = TemporalSegDataset(
        split_names=tuple(d.get("train_splits", ["train"])),
        max_sequences=d.get("max_train_sequences"),
        **common,
    )
    val_ds = TemporalSegDataset(
        split_names=tuple(d.get("val_splits", ["validation"])),
        max_sequences=d.get("max_val_sequences"),
        **{**common, "augmentation": {"enabled": False}},
    )
    tr = DataLoader(train_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=True, num_workers=int(cfg["training"].get("num_workers", 4)), pin_memory=True, collate_fn=collate_seg)
    va = DataLoader(val_ds, batch_size=int(cfg["training"].get("val_batch_size", cfg["training"]["batch_size"])), shuffle=False, num_workers=int(cfg["training"].get("num_workers", 4)), pin_memory=True, collate_fn=collate_seg)
    return train_ds, val_ds, tr, va


def seg_step(model, batch, device, cfg):
    rgb = batch["rgb"].to(device, non_blocking=True)
    outputs = model(rgb, return_all_frames=True)
    loss, stats = segmentation_loss(outputs, batch["targets"], cfg.get("loss", {}))
    return loss, stats


@torch.no_grad()
def seg_val(model, loader, device, cfg):
    model.eval()
    vals = []
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        outputs = model(rgb, return_all_frames=True)
        loss, _ = segmentation_loss(outputs, batch["targets"], cfg.get("loss", {}))
        vals.append(float(loss))
    v = sum(vals) / max(1, len(vals))
    return v, {"val_loss": v}


def build_keypoint_loaders(cfg: dict[str, Any]):
    d = cfg["dataset"]
    common = dict(
        root=d["root"],
        window_size=int(cfg["model"].get("window_size", 5)),
        crop_size=tuple(cfg["model"].get("crop_size", [192, 192])),
        crop_padding=float(cfg["model"].get("crop_padding", 0.35)),
        seed=int(cfg.get("seed", 42)),
    )
    train_ds = GateKeypointDataset(split_names=tuple(d.get("train_splits", ["train"])), max_sequences=d.get("max_train_sequences"), augmentation=cfg.get("augmentation"), **common)
    val_ds = GateKeypointDataset(split_names=tuple(d.get("val_splits", ["validation"])), max_sequences=d.get("max_val_sequences"), augmentation={"enabled": False}, **common)
    tr = DataLoader(train_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=True, num_workers=int(cfg["training"].get("num_workers", 4)), pin_memory=True, collate_fn=collate_keypoints)
    va = DataLoader(val_ds, batch_size=int(cfg["training"].get("val_batch_size", cfg["training"]["batch_size"])), shuffle=False, num_workers=int(cfg["training"].get("num_workers", 4)), pin_memory=True, collate_fn=collate_keypoints)
    return train_ds, val_ds, tr, va


def keypoint_step(model, batch, device, cfg):
    x = batch["x"].to(device, non_blocking=True)
    outputs = model(x)
    moved = {"keypoints": batch["keypoints"], "valid": batch["valid"], "visibility": batch["visibility"]}
    loss, stats = keypoint_loss(outputs, moved, cfg.get("loss", {}))
    return loss, stats


@torch.no_grad()
def keypoint_val(model, loader, device, cfg):
    model.eval()
    vals = []
    px_errs = []
    crop_w, crop_h = map(float, cfg["model"].get("crop_size", [192, 192]))
    scale = torch.tensor([crop_w, crop_h], device=device)
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        outputs = model(x)
        loss, _ = keypoint_loss(outputs, batch, cfg.get("loss", {}))
        vals.append(float(loss))
        gt = batch["keypoints"].to(device)
        valid = batch["valid"].to(device)
        if valid.any():
            err = torch.linalg.vector_norm((outputs["keypoints"] - gt) * scale, dim=-1)
            px_errs.extend(err[valid].detach().cpu().tolist())
    v = sum(vals) / max(1, len(vals))
    mean_px = sum(px_errs) / max(1, len(px_errs))
    return v, {"val_loss": v, "val_mean_keypoint_error_px_crop": mean_px}
