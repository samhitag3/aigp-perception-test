"""Shared canonical training utilities for GatePoseNet-MG."""
from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from gatenet.schedule import build_scheduler
from gatenet.utils import AverageMeter, save_checkpoint
from .canonical_dataset_mg import CanonicalGateSequenceDatasetMG
from .data_sources import MultiSourceCanonicalDataset, canonical_source_specs


def set_manifest(cfg: dict, kind: str, value: str) -> None:
    key = {"train": "train_manifest", "validation": "val_manifest", "test": "test_manifest"}[kind]
    cfg.setdefault("data", {})[key] = value
    for src in cfg["data"].get("sources", []):
        src[key] = value


def _manifest_for(spec, split: str) -> str:
    return {"train": spec.train_manifest,
            "validation": spec.val_manifest,
            "val": spec.val_manifest,
            "test": spec.test_manifest}[split]


def build_mg_dataset(cfg: dict, split: str, ego_dropout: float = 0.0):
    d = cfg["data"]
    specs = canonical_source_specs(d)
    if not specs:
        raise ValueError("MG canonical config requires data.dataset_root or data.sources")
    stride = int(d.get("stride", 4) if split == "train" else
                 d.get("val_stride", d.get("window", 8)) if split in {"validation", "val"} else
                 d.get("test_stride", d.get("window", 8)))
    max_gates = int(d.get("max_gates", cfg.get("model", {}).get("n_queries", 8)))
    n_queries = int(cfg.get("model", {}).get("n_queries", 8))
    if max_gates > n_queries:
        raise ValueError(f"data.max_gates={max_gates} cannot exceed model.n_queries={n_queries}")
    size = (int(d.get("height", 192)), int(d.get("width", 320)))
    datasets = [CanonicalGateSequenceDatasetMG(
        s.root, _manifest_for(s, split), window=int(d.get("window", 8)),
        stride=stride, size=size,
        pose_sup_max_m=float(d.get("pose_sup_max_m", 25.0)),
        nominal_focal=float(d.get("nominal_focal", 320.0)),
        ego_dropout=float(ego_dropout), max_gates=max_gates,
    ) for s in specs]
    if len(datasets) == 1:
        return datasets[0]
    return MultiSourceCanonicalDataset(datasets, specs)


def make_loaders(cfg: dict):
    d, t = cfg["data"], cfg["train"]
    train_ds = build_mg_dataset(cfg, "train", ego_dropout=float(d.get("ego_dropout", 0.25)))
    val_ds = build_mg_dataset(cfg, "validation", ego_dropout=0.0)
    nw = int(t.get("num_workers", 8))
    common = dict(num_workers=nw, pin_memory=True)
    if nw > 0:
        common["persistent_workers"] = True
    bs = int(t.get("batch_size", 12))
    tl = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True, **common)
    vl = DataLoader(val_ds, batch_size=bs, shuffle=False, drop_last=False, **common)
    return train_ds, val_ds, tl, vl


@torch.no_grad()
def validate_mg(model, loader, criterion, device):
    model.eval()
    lm = AverageMeter()
    tp_err, iou_all, pres_ok = [], [], []
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        pred = model(batch["image"], batch["ego"])
        loss, _ = criterion(pred, batch)
        lm.update(float(loss), n=batch["image"].shape[0])

        q_sel = pred["target_logit"].argmax(dim=2)
        pos_q = pred["position"].gather(
            2, q_sel[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
        t_idx = batch["g_target"].argmax(dim=2)
        pos_gt = batch["g_position"].gather(
            2, t_idx[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
        sel = ((batch["g_target"].max(dim=2).values > 0.5)
               & (batch["frame_ok"] > 0.5)
               & (batch["g_visible"].gather(2, t_idx[..., None]).squeeze(-1) > 0.5))
        if sel.any():
            tp_err.append((pos_q - pos_gt).norm(dim=-1)[sel].cpu())

        prob = torch.sigmoid(pred["presence_logit"])
        n_pred = (prob > 0.5).sum(dim=2).float()
        n_gt = batch["g_valid"].sum(dim=2)
        valid_frames = batch["frame_ok"] > 0.5
        if valid_frames.any():
            pres_ok.append((n_pred == n_gt)[valid_frames].float().cpu())

        pm = torch.sigmoid(pred["mask_logit"]) > 0.5
        for b in range(pm.shape[0]):
            for tt in range(pm.shape[1]):
                if batch["frame_ok"][b, tt] < 0.5:
                    continue
                for g in range(batch["g_valid"].shape[2]):
                    if batch["g_valid"][b, tt, g] < 0.5 or batch["g_mask"][b, tt, g].sum() < 4:
                        continue
                    gm = batch["g_mask"][b, tt, g] > 0.5
                    inter = (pm[b, tt] & gm).sum(dim=(-1, -2)).float()
                    union = (pm[b, tt] | gm).sum(dim=(-1, -2)).float() + 1e-6
                    iou_all.append(float((inter / union).max()))
    out = {"val_loss": lm.avg}
    out["target_pos_err_m"] = float(torch.cat(tp_err).median()) if tp_err else float("inf")
    out["mask_iou"] = sum(iou_all) / len(iou_all) if iou_all else 0.0
    out["presence_acc"] = float(torch.cat(pres_ok).mean()) if pres_ok else 0.0
    return out


def fit_mg(cfg, model, train_loader, val_loader, criterion, optimizer, scheduler,
           device, out_dir: Path, epochs: int, tag: str = "mg") -> float:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        raise RuntimeError("tensorboard is required for training; run `uv sync` first") from exc
    tb = SummaryWriter(str(out_dir / "tb"))
    amp = device.type == "cuda" and bool(cfg["train"].get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best = float("inf")
    step = 0
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))
    try:
        for epoch in range(epochs):
            model.train(); meter = AverageMeter(); t0 = time.time()
            for batch in tqdm(train_loader, desc=f"[{tag}] epoch {epoch+1}/{epochs}", leave=False):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=amp):
                    pred = model(batch["image"], batch["ego"])
                    loss, logs = criterion(pred, batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer); scaler.update()
                meter.update(float(loss), n=batch["image"].shape[0]); step += 1
                if step % int(cfg["train"].get("log_interval", 20)) == 0:
                    for k, v in logs.items():
                        tb.add_scalar(f"train/{k}", v, step)
            metrics = validate_mg(model, val_loader, criterion, device)
            for k, v in metrics.items():
                tb.add_scalar(f"val/{k}", v, epoch)
            if scheduler is not None:
                scheduler.step()
            print(f"[{tag}] epoch {epoch+1}/{epochs} train_loss={meter.avg:.4f} "
                  f"val_loss={metrics['val_loss']:.4f} target_pos={metrics['target_pos_err_m']:.3f}m "
                  f"mask_iou={metrics['mask_iou']:.3f} presence_acc={metrics['presence_acc']:.3f} "
                  f"({time.time()-t0:.0f}s)")
            sel = metrics["target_pos_err_m"]
            extra = {"config": cfg, "val_metrics": metrics}
            if sel < best:
                best = sel
                save_checkpoint(out_dir / "best.pt", model, optimizer, scheduler,
                                epoch=epoch, best_metric=best, extra=extra)
            save_checkpoint(out_dir / "last.pt", model, optimizer, scheduler,
                            epoch=epoch, best_metric=best, extra=extra)
    finally:
        tb.close()
    return best


def make_optimizer(cfg, model):
    return torch.optim.AdamW(model.parameters(), lr=float(cfg["train"].get("lr", 7e-4)),
                             weight_decay=float(cfg["train"].get("weight_decay", 0.01)))


def make_scheduler(cfg, optimizer):
    return build_scheduler(optimizer, cfg.get("schedule"))