"""Common binary-mask segmentation metrics (numpy)."""

from __future__ import annotations

import numpy as np


def mask_metrics(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> dict:
    """IoU / Dice / precision / recall / f1 / accuracy for two binary masks."""
    p = (pred > 0).reshape(-1).astype(np.float64)
    g = (gt > 0).reshape(-1).astype(np.float64)
    tp = float((p * g).sum())
    fp = float((p * (1 - g)).sum())
    fn = float(((1 - p) * g).sum())
    tn = float(((1 - p) * (1 - g)).sum())
    return {
        "iou": (tp + eps) / (tp + fp + fn + eps),
        "dice": (2 * tp + eps) / (2 * tp + fp + fn + eps),
        "precision": (tp + eps) / (tp + fp + eps),
        "recall": (tp + eps) / (tp + fn + eps),
        "f1": (2 * tp + eps) / (2 * tp + fp + fn + eps),
        "accuracy": (tp + tn + eps) / (tp + tn + fp + fn + eps),
    }


class MetricAccumulator:
    """Accumulate per-image metric dicts into running means."""

    def __init__(self):
        self.sums = {}
        self.n = 0

    def update(self, metrics: dict):
        for k, v in metrics.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v)
        self.n += 1

    def result(self) -> dict:
        if self.n == 0:
            return {}
        return {k: v / self.n for k, v in self.sums.items()}
