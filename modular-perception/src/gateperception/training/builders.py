from __future__ import annotations
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from gateperception.data.canonical import (
    TemporalSegDataset,
    GateKeypointDataset,
    collate_seg,
    collate_keypoints,
)
from gateperception.training.seg_losses import segmentation_loss
from gateperception.training.keypoint_losses import keypoint_loss


def _dataset_sources(d: dict[str, Any]) -> list[dict[str, str]]:
    """Return normalized dataset sources while preserving single-root compatibility."""
    if d.get("sources"):
        out: list[dict[str, str]] = []
        for i, src in enumerate(d["sources"]):
            if isinstance(src, str):
                out.append({"name": f"source_{i}", "root": src})
            else:
                out.append({"name": str(src.get("name", f"source_{i}")), "root": str(src["root"])})
        return out
    if "root" not in d:
        raise KeyError("dataset must define either 'root' or non-empty 'sources'")
    return [{"name": str(d.get("name", "dataset")), "root": str(d["root"])}]


def _combine(parts: list[Dataset]) -> Dataset:
    if not parts:
        raise ValueError("No dataset sources configured")
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)


def _loader(ds: Dataset, *, batch_size: int, shuffle: bool, num_workers: int, collate_fn, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def build_seg_loaders(cfg: dict[str, Any]):
    d = cfg["dataset"]
    seed = int(cfg.get("seed", 42))
    selection_seed = int(d.get("selection_seed", seed))
    train_fraction = float(d.get("train_fraction", 1.0))
    val_fraction = float(d.get("val_fraction", 1.0))
    sources = _dataset_sources(d)
    val_spec = dict(d)
    if d.get("validation_sources"):
        val_spec["sources"] = d["validation_sources"]
        val_spec.pop("root", None)
    val_sources = _dataset_sources(val_spec)

    train_parts: list[Dataset] = []
    val_parts: list[Dataset] = []
    for src in sources:
        common = dict(
            root=src["root"],
            window_size=int(cfg["model"].get("window_size", 4)),
            width=int(d.get("width", 640)),
            height=int(d.get("height", 360)),
            seed=seed,
            selection_seed=selection_seed,
        )
        train_parts.append(
            TemporalSegDataset(
                split_names=tuple(d.get("train_splits", ["train"])),
                max_sequences=d.get("max_train_sequences"),
                sequence_fraction=train_fraction,
                augmentation=cfg.get("augmentation"),
                **common,
            )
        )
    for src in val_sources:
        val_common = dict(
            root=src["root"],
            window_size=int(cfg["model"].get("window_size", 4)),
            width=int(d.get("width", 640)),
            height=int(d.get("height", 360)),
            seed=seed,
            selection_seed=selection_seed,
        )
        val_parts.append(
            TemporalSegDataset(
                split_names=tuple(d.get("val_splits", ["validation"])),
                max_sequences=d.get("max_val_sequences"),
                sequence_fraction=val_fraction,
                augmentation={"enabled": False},
                **val_common,
            )
        )

    train_ds = _combine(train_parts)
    val_ds = _combine(val_parts)
    nworkers = int(cfg["training"].get("num_workers", 4))
    tr = _loader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=nworkers,
        collate_fn=collate_seg,
        seed=seed,
    )
    va = _loader(
        val_ds,
        batch_size=int(cfg["training"].get("val_batch_size", cfg["training"]["batch_size"])),
        shuffle=False,
        num_workers=nworkers,
        collate_fn=collate_seg,
        seed=seed,
    )
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
    seed = int(cfg.get("seed", 42))
    selection_seed = int(d.get("selection_seed", seed))
    train_fraction = float(d.get("train_fraction", 1.0))
    val_fraction = float(d.get("val_fraction", 1.0))
    sources = _dataset_sources(d)
    val_spec = dict(d)
    if d.get("validation_sources"):
        val_spec["sources"] = d["validation_sources"]
        val_spec.pop("root", None)
    val_sources = _dataset_sources(val_spec)

    train_parts: list[Dataset] = []
    val_parts: list[Dataset] = []
    for src in sources:
        common = dict(
            root=src["root"],
            window_size=int(cfg["model"].get("window_size", 5)),
            crop_size=tuple(cfg["model"].get("crop_size", [192, 192])),
            crop_padding=float(cfg["model"].get("crop_padding", 0.35)),
            seed=seed,
            selection_seed=selection_seed,
        )
        train_parts.append(
            GateKeypointDataset(
                split_names=tuple(d.get("train_splits", ["train"])),
                max_sequences=d.get("max_train_sequences"),
                sequence_fraction=train_fraction,
                augmentation=cfg.get("augmentation"),
                **common,
            )
        )
    for src in val_sources:
        val_common = dict(
            root=src["root"],
            window_size=int(cfg["model"].get("window_size", 5)),
            crop_size=tuple(cfg["model"].get("crop_size", [192, 192])),
            crop_padding=float(cfg["model"].get("crop_padding", 0.35)),
            seed=seed,
            selection_seed=selection_seed,
        )
        val_parts.append(
            GateKeypointDataset(
                split_names=tuple(d.get("val_splits", ["validation"])),
                max_sequences=d.get("max_val_sequences"),
                sequence_fraction=val_fraction,
                augmentation={"enabled": False},
                **val_common,
            )
        )

    train_ds = _combine(train_parts)
    val_ds = _combine(val_parts)
    nworkers = int(cfg["training"].get("num_workers", 4))
    tr = _loader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=nworkers,
        collate_fn=collate_keypoints,
        seed=seed,
    )
    va = _loader(
        val_ds,
        batch_size=int(cfg["training"].get("val_batch_size", cfg["training"]["batch_size"])),
        shuffle=False,
        num_workers=nworkers,
        collate_fn=collate_keypoints,
        seed=seed,
    )
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
