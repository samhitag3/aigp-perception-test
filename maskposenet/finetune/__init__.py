"""Unified fine-tuning package for gate segmentation.

Backends:
  * ``gatenet`` — the in-repo lightweight U-Net (gatenet/).
  * ``yolo26``  — Ultralytics YOLO26 instance segmentation.
  * ``yoloe``   — Ultralytics YOLOE open-vocabulary segmentation.

All backends consume datasets produced by the sam3-autolabeler and imported into
``/Users/c.k./gateNET-1/data`` via ``finetune.data`` (see scripts/import_data.py).
"""

from .registry import BACKENDS, resolve

__all__ = ["BACKENDS", "resolve"]
