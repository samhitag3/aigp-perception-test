"""GateNet: lightweight U-Net gate segmentation for autonomous drone racing.

Faithful re-implementation of the GateNet described in:
  * Bahnam et al., "MonoRace: Winning Champion-Level Drone Racing with Robust
    Monocular AI" (A2RL 2025) — authoritative training recipe (Appendix /
    Perception > GateNet).
  * Verraest et al., "SkyDreamer" (arXiv:2510.14783) — reuses the same network
    (Appendix A).
"""

from .model import GateNet, GateNetInference, build_gatenet
from .losses import DeepSupervisionLoss, dice_loss, bce_loss
from .metrics import seg_metrics

__all__ = [
    "GateNet",
    "GateNetInference",
    "build_gatenet",
    "DeepSupervisionLoss",
    "dice_loss",
    "bce_loss",
    "seg_metrics",
]
