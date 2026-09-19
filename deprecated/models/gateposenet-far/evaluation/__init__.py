"""Evaluation package: compare GateNet and YOLO26/YOLOE on the same gate test set.

All backends are scored on a common binary-mask metric (IoU / Dice / precision /
recall / F1) over data/test, so GateNet and the YOLO models are directly
comparable. YOLO models additionally report Ultralytics' native box/mask mAP.
"""

from .metrics import MetricAccumulator, mask_metrics

__all__ = ["MetricAccumulator", "mask_metrics"]
