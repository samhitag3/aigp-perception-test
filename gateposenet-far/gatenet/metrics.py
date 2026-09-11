"""Segmentation metrics. IoU is the primary metric used by MonoRace."""

from __future__ import annotations

import torch


@torch.no_grad()
def seg_metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5,
                eps: float = 1e-6) -> dict:
    """Compute IoU / Dice / precision / recall / accuracy for a batch.

    Args:
        logits: raw logits (B,1,H,W) — the highest-res GateNet output.
        target: binary GT (B,1,H,W).
    Returns dict of python floats (batch means).
    """
    pred = (torch.sigmoid(logits) > threshold).float().flatten(1)
    tgt = (target > 0.5).float().flatten(1)

    tp = (pred * tgt).sum(1)
    fp = (pred * (1 - tgt)).sum(1)
    fn = ((1 - pred) * tgt).sum(1)
    tn = ((1 - pred) * (1 - tgt)).sum(1)

    iou = (tp + eps) / (tp + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    prec = (tp + eps) / (tp + fp + eps)
    rec = (tp + eps) / (tp + fn + eps)
    acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    return {
        "iou": float(iou.mean()),
        "dice": float(dice.mean()),
        "precision": float(prec.mean()),
        "recall": float(rec.mean()),
        "accuracy": float(acc.mean()),
    }
