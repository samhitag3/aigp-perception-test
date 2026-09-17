from __future__ import annotations
from typing import Any
import torch
from torch import nn
import torch.nn.functional as F

from .common import ConvGRU


class TemporalMaskKeypointNet(nn.Module):
    """Mask-conditioned temporal keypoint network.

    Input per gate is [B,T,4,H,W]: RGB plus that gate's binary instance mask.
    It predicts all eight amodal/projected keypoints plus visibility state.
    """

    def __init__(self, hidden_dim: int = 128, num_visibility_classes: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 5, stride=2, padding=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(96, hidden_dim, 3, stride=2, padding=1), nn.BatchNorm2d(hidden_dim), nn.ReLU(),
        )
        self.temporal = ConvGRU(hidden_dim, hidden_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.coord_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 16))
        self.visibility_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 8 * num_visibility_classes)
        )
        self.num_visibility_classes = num_visibility_classes

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b, t, c, h, w = x.shape
        feats = []
        for i in range(t):
            feats.append(self.encoder(x[:, i]))
        last, _ = self.temporal(torch.stack(feats, dim=1))
        z = self.pool(last).flatten(1)
        # Do not clamp: out-of-frame/amodal keypoints can legitimately lie outside [0,1] crop coordinates.
        keypoints = self.coord_head(z).view(b, 8, 2)
        visibility_logits = self.visibility_head(z).view(b, 8, self.num_visibility_classes)
        return {"keypoints": keypoints, "visibility_logits": visibility_logits}


def build_keypoint_model(cfg: dict[str, Any]) -> TemporalMaskKeypointNet:
    m = cfg.get("model", cfg)
    return TemporalMaskKeypointNet(hidden_dim=int(m.get("hidden_dim", 128)))
