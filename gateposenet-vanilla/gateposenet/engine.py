"""Training / validation loop for GatePoseNet sequence windows.

Mirrors gatenet/engine.py conventions (AMP, tqdm, TensorBoard, best/last
checkpoints) but consumes dict batches of (B, T, ...) windows and selects the
best checkpoint by symmetry-aware VISIBLE-frame relative position error.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatenet.utils import AverageMeter, save_checkpoint

from .dataset import GateSequenceDataset
from .canonical_dataset import CanonicalGateSequenceDataset
from .data_sources import build_canonical_dataset, canonical_source_specs
from .metrics import merge_metric_acc, pose_metrics, reduce_metrics


def build_datasets(cfg: dict):
    """Build either the legacy dataset loader or the canonical-contract adapter.

    Presence of ``data.dataset_root`` selects the canonical adapter. This keeps
    old configs/checkpoints working while allowing the unchanged GatePoseNet
    architecture to train directly from the standardized source-of-truth data.
    """
    d = cfg["data"]
    size = (int(d.get("height", 192)), int(d.get("width", 320)))
    common = dict(
        window=int(d.get("window", 8)),
        size=size,
        pose_sup_max_m=float(d.get("pose_sup_max_m", 20.0)),
        nominal_focal=float(d.get("nominal_focal", 320.0)),
    )
    if d.get("dataset_root") or d.get("sources"):
        train_ds = build_canonical_dataset(
            d, "train", stride=int(d.get("stride", 4)),
            ego_dropout=float(d.get("ego_dropout", 0.25)))
        val_ds = build_canonical_dataset(
            d, "validation", stride=int(d.get("val_stride", 8)),
            ego_dropout=0.0)
        return train_ds, val_ds

    train_ds = GateSequenceDataset(
        d["train_roots"], stride=int(d.get("stride", 4)),
        ego_dropout=float(d.get("ego_dropout", 0.25)), **common)
    val_ds = GateSequenceDataset(
        d["val_roots"], stride=int(d.get("val_stride", 8)),
        ego_dropout=0.0, **common)
    return train_ds, val_ds


def make_loaders(cfg: dict, train_ds, val_ds):
    t = cfg["train"]
    kw = dict(num_workers=int(t.get("num_workers", 4)), pin_memory=True,
              persistent_workers=int(t.get("num_workers", 4)) > 0)
    train_loader = DataLoader(train_ds, batch_size=int(t["batch_size"]),
                              shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=int(t["batch_size"]),
                            shuffle=False, **kw)
    return train_loader, val_loader


def make_optimizer(cfg: dict, model) -> torch.optim.Optimizer:
    t = cfg["train"]
    name = str(t.get("optimizer", "adamw")).lower()
    lr = float(t.get("lr", 1e-3))
    wd = float(t.get("weight_decay", 0.01))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                               weight_decay=wd)
    raise ValueError(f"unknown optimizer {name!r}")


def _to_device(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def validate(model, loader, criterion, device, img_wh) -> tuple[dict, dict]:
    model.eval()
    loss_meter = AverageMeter()
    logs_sum: dict[str, AverageMeter] = {}
    acc: dict = {}
    for batch in loader:
        batch = _to_device(batch, device)
        pred = model(batch["image"], batch["ego"])
        loss, logs = criterion(pred, batch)
        loss_meter.update(float(loss), n=batch["image"].shape[0])
        for k, v in logs.items():
            logs_sum.setdefault(k, AverageMeter()).update(v)
        merge_metric_acc(acc, pose_metrics(pred, batch, img_wh))
    metrics = reduce_metrics(acc)
    metrics["val_loss"] = loss_meter.avg
    return metrics, {k: m.avg for k, m in logs_sum.items()}


def fit(cfg, model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, out_dir: Path, epochs: int, start_epoch: int = 0,
        best_metric: float = float("inf"), tag: str = "gatepose") -> float:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Never silently destroy a previous run's best checkpoint: archive it
    # with its mtime stamp before this run's first save can overwrite it.
    prev = out_dir / "best.pt"
    if start_epoch == 0 and prev.exists():
        import datetime as _dt
        stamp = _dt.datetime.fromtimestamp(
            prev.stat().st_mtime).strftime("%Y-%m-%d_%H%M")
        archive = out_dir / f"best_prev_{stamp}.pt"
        if not archive.exists():
            import shutil
            shutil.copy2(prev, archive)
            print(f"[{tag}] archived previous best.pt -> {archive.name}")
    tb = SummaryWriter(str(out_dir / "tb"))
    img_wh = (int(cfg["data"].get("native_width", 640)),
              int(cfg["data"].get("native_height", 360)))
    amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    log_every = int(cfg["train"].get("log_interval", 20))
    clip = float(cfg["train"].get("grad_clip", 1.0))

    step = start_epoch * len(train_loader)
    for epoch in range(start_epoch, epochs):
        model.train()
        meter = AverageMeter()
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"[{tag}] epoch {epoch + 1}/{epochs}",
                    leave=False)
        for batch in pbar:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                pred = model(batch["image"], batch["ego"])
                loss, logs = criterion(pred, batch)
            scaler.scale(loss).backward()
            if clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            meter.update(float(loss), n=batch["image"].shape[0])
            step += 1
            if step % log_every == 0:
                for k, v in logs.items():
                    tb.add_scalar(f"train/{k}", v, step)
                pbar.set_postfix(loss=f"{meter.avg:.4f}")

        metrics, val_logs = validate(model, val_loader, criterion, device,
                                     img_wh)
        for k, v in metrics.items():
            tb.add_scalar(f"val/{k}", v, epoch)
        for k, v in val_logs.items():
            tb.add_scalar(f"val_loss/{k}", v, epoch)
        if scheduler is not None:
            scheduler.step()

        # selection metric with a fallback so best.pt is still written when
        # no visible frames produced pos_err_rel (e.g. tiny smoke datasets).
        sel = metrics.get("pos_err_rel", metrics.get("val_loss", float("inf")))
        dt = time.time() - t0
        print(f"[{tag}] epoch {epoch + 1}/{epochs} "
              f"train_loss={meter.avg:.4f} val_loss={metrics['val_loss']:.4f} "
              f"pos_rel={metrics.get('pos_err_rel', float('nan')):.4f} "
              f"pos_m={metrics.get('pos_err_m', float('nan')):.3f} "
              f"rot={metrics.get('rot_err_deg', float('nan')):.2f}deg "
              f"normal={metrics.get('normal_err_deg', float('nan')):.2f}deg "
              f"center={metrics.get('center_err_px', float('nan')):.1f}px "
              f"vis_acc={metrics.get('vis_acc', float('nan')):.3f} "
              f"({dt:.0f}s)")

        extra = {"config": cfg, "val_metrics": metrics}
        # update best BEFORE writing last.pt so a resume from last.pt cannot
        # later overwrite best.pt with a worse checkpoint.
        if sel < best_metric:
            best_metric = sel
            save_checkpoint(out_dir / "best.pt", model, optimizer, scheduler,
                            epoch=epoch, best_metric=best_metric, extra=extra)
        save_checkpoint(out_dir / "last.pt", model, optimizer, scheduler,
                        epoch=epoch, best_metric=best_metric, extra=extra)
    tb.close()
    return best_metric
