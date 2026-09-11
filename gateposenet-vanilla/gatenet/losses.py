"""GateNet losses.

Per the MonoRace recipe, every output map is supervised with

    L_i = Dice(y_i, y_hat_i) + 2 * BCE(y_i, y_hat_i)

and the total loss applies output-specific scaling that emphasises the
higher-resolution maps:

    L_tot = 4*L_0 + 2*L_1 + L_2 + L_3 + L_4
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    """Soft Dice loss for binary masks. logits and target are (B,1,H,W)."""
    prob = torch.sigmoid(logits).flatten(1)
    tgt = target.flatten(1)
    inter = (prob * tgt).sum(1)
    union = prob.sum(1) + tgt.sum(1)
    dice = (2.0 * inter + eps) / (union + eps)
    return (1.0 - dice).mean()


def bce_loss(logits: torch.Tensor, target: torch.Tensor):
    """Binary cross-entropy (with logits)."""
    return F.binary_cross_entropy_with_logits(logits, target)


class DeepSupervisionLoss(nn.Module):
    """Multi-scale Dice + BCE loss for GateNet's 5 output maps.

    Args:
        scale_weights: per-output weight (y0..y4). Default (4,2,1,1,1).
        bce_weight: weight on the BCE term within each L_i. Default 2.0.
    """

    def __init__(self, scale_weights=(4.0, 2.0, 1.0, 1.0, 1.0), bce_weight: float = 2.0):
        super().__init__()
        self.scale_weights = tuple(scale_weights)
        self.bce_weight = float(bce_weight)

    def forward(self, outputs, target):
        """outputs: list of logit maps (highest-res first); target: (B,1,H,W)."""
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        total = outputs[0].new_zeros(())
        logs = {}
        for i, logit in enumerate(outputs):
            tgt = target
            if logit.shape[-2:] != target.shape[-2:]:
                # area interpolation then re-binarise the down-scaled GT mask
                tgt = F.interpolate(target, size=logit.shape[-2:], mode="area")
                tgt = (tgt > 0.5).float()
            li = dice_loss(logit, tgt) + self.bce_weight * bce_loss(logit, tgt)
            w = self.scale_weights[i] if i < len(self.scale_weights) else 1.0
            total = total + w * li
            logs[f"loss/scale{i}"] = float(li.detach())
        logs["loss/total"] = float(total.detach())
        return total, logs
