from __future__ import annotations
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

from .common import ConvGRU


class ResNet18Features(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        m = resnet18(weights=weights)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1  # 1/4, 64
        self.layer2 = m.layer2  # 1/8, 128
        self.layer3 = m.layer3  # 1/16, 256

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        return c2, c3, c4


class TemporalGateInstanceSegmenter(nn.Module):
    """Temporal query-based instance segmenter.

    A ConvGRU maintains visual memory over the input frame window. Fixed queries decode
    gate candidates from the temporally fused feature map; each query emits gate/no-gate
    confidence, a normalized box, a mask embedding, and a track embedding.
    """

    def __init__(
        self,
        num_queries: int = 8,
        hidden_dim: int = 128,
        track_dim: int = 64,
        decoder_layers: int = 3,
        nheads: int = 4,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.num_queries = int(num_queries)
        self.hidden_dim = int(hidden_dim)
        self.backbone = ResNet18Features(pretrained=pretrained_backbone)
        self.proj_c4 = nn.Conv2d(256, hidden_dim, 1)
        self.temporal = ConvGRU(hidden_dim, hidden_dim)

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=decoder_layers)

        self.object_head = nn.Linear(hidden_dim, 1)
        self.box_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 4))
        self.mask_embed = nn.Linear(hidden_dim, hidden_dim)
        self.track_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, track_dim))

        # 1/4-resolution mask feature pyramid from current RGB feature + temporal state.
        self.c4_to_c3 = nn.Conv2d(hidden_dim, 128, 1)
        self.c3_lat = nn.Conv2d(128, 128, 1)
        self.c2_lat = nn.Conv2d(64, 128, 1)
        self.mask_features = nn.Sequential(nn.Conv2d(128, hidden_dim, 3, padding=1), nn.ReLU(), nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1))

    def _decode_one(self, memory_map: torch.Tensor, c2: torch.Tensor, c3: torch.Tensor) -> dict[str, torch.Tensor]:
        b, c, h, w = memory_map.shape
        memory = memory_map.flatten(2).transpose(1, 2)  # [B,HW,C]
        q = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        z = self.decoder(q, memory)
        object_logits = self.object_head(z).squeeze(-1)
        boxes = torch.sigmoid(self.box_head(z))
        # Force xyxy ordering without losing differentiability.
        xy0 = torch.minimum(boxes[..., :2], boxes[..., 2:])
        xy1 = torch.maximum(boxes[..., :2], boxes[..., 2:])
        boxes = torch.cat([xy0, xy1], dim=-1)

        p3 = self.c3_lat(c3) + F.interpolate(self.c4_to_c3(memory_map), size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.c2_lat(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        mask_features = self.mask_features(p2)
        emb = self.mask_embed(z)
        mask_logits = torch.einsum("bqc,bchw->bqhw", emb, mask_features)
        track_embeddings = F.normalize(self.track_head(z), dim=-1)
        return {
            "object_logits": object_logits,
            "boxes": boxes,
            "mask_logits": mask_logits,
            "track_embeddings": track_embeddings,
        }

    def forward(self, rgb: torch.Tensor, return_all_frames: bool = False) -> dict[str, Any]:
        # rgb [B,T,3,H,W]
        b, t, _, _, _ = rgb.shape
        c2s: list[torch.Tensor] = []
        c3s: list[torch.Tensor] = []
        c4s: list[torch.Tensor] = []
        for i in range(t):
            c2, c3, c4 = self.backbone(rgb[:, i])
            c2s.append(c2)
            c3s.append(c3)
            c4s.append(self.proj_c4(c4))
        temporal_input = torch.stack(c4s, dim=1)
        _, states = self.temporal(temporal_input)
        if return_all_frames:
            frames = [self._decode_one(states[i], c2s[i], c3s[i]) for i in range(t)]
        else:
            frames = [self._decode_one(states[-1], c2s[-1], c3s[-1])]
        return {"frames": frames}


def build_segmentation_model(cfg: dict[str, Any]) -> TemporalGateInstanceSegmenter:
    m = cfg.get("model", cfg)
    return TemporalGateInstanceSegmenter(
        num_queries=int(m.get("num_queries", 8)),
        hidden_dim=int(m.get("hidden_dim", 128)),
        track_dim=int(m.get("track_dim", 64)),
        decoder_layers=int(m.get("decoder_layers", 3)),
        nheads=int(m.get("nheads", 4)),
        pretrained_backbone=bool(m.get("pretrained_backbone", True)),
    )
