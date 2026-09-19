from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Callable
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_metric: float, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best_metric": best_metric, "config": cfg}, path)


def train_loop(
    model,
    train_loader,
    val_loader,
    optimizer,
    epochs: int,
    device: torch.device,
    run_dir: Path,
    cfg: dict,
    step_fn: Callable,
    val_fn: Callable,
    amp: bool = True,
    scheduler=None,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    best = float("inf")
    history = []
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        totals = []
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                loss, stats = step_fn(model, batch, device, cfg)
            scaler.scale(loss).backward()
            max_norm = float(cfg.get("training", {}).get("grad_clip_norm", 1.0))
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            totals.append(float(loss.detach()))
            pbar.set_postfix(loss=f"{totals[-1]:.4f}")
        val_metric, val_stats = val_fn(model, val_loader, device, cfg)
        if scheduler is not None:
            scheduler.step()
        row = {"epoch": epoch, "train_loss": sum(totals) / max(1, len(totals)), "val_metric": val_metric, **val_stats}
        history.append(row)
        with open(run_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, best, cfg)
        if val_metric < best:
            best = val_metric
            save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, best, cfg)
        print(json.dumps(row))
    return {"best_metric": best, "training_time_seconds": time.time() - start, "history": history}
