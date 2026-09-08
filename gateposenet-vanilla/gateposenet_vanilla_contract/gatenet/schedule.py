"""Learning-rate schedule used to train GateNet.

MonoRace: base LR 1e-3, multiplied by sqrt(0.1) at epochs 10, 33, 66 and 90,
giving a final LR of 1e-5 (four drops -> 0.1^2 = 1e-2 factor).
"""

from __future__ import annotations

import math

import torch


SQRT_0_1 = math.sqrt(0.1)  # ~0.31623


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict | None = None):
    """MultiStepLR with gamma = sqrt(0.1) at the MonoRace milestones."""
    cfg = cfg or {}
    milestones = cfg.get("milestones", [10, 33, 66, 90])
    gamma = float(cfg.get("gamma", SQRT_0_1))
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(milestones), gamma=gamma
    )
