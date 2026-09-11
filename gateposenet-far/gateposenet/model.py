"""GatePoseNet-MG — multi-gate: per-gate masks + keypoints + pose.

Multi-input, multi-output. Same lightweight temporal backbone as GatePoseNet
(GateNet encoder -> ego-motion [v, omega, dt] embedding -> ConvGRU state),
with the single-target MLP head replaced by a SMALL TRANSFORMER DECODER
(DETR/MaskFormer-style):

* N learned gate queries self-attend and cross-attend to the ConvGRU state
  (12x20 = 240 tokens + 2-D sine positional encoding);
* every query emits ITS OWN gate: presence logit, target score ("the gate I
  am flying at"), 4 corners + inside flags, center, metric position,
  log-depth, 6-D rotation, visible fraction — and an instance MASK via the
  MaskFormer mechanism (query mask-embedding dotted with per-pixel decoder
  features at 1/4 resolution);
* training matches queries to per-gate ground truth with Hungarian
  assignment (losses_mg.py); at deployment the per-gate keypoints feed PnP
  per gate (the model is the estimator; PnP checks/corrects).

The temporal state still carries every gate through blind stretches;
`step(image, ego, h)` remains the 90-120 Hz deployment API.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatenet.model import BASE_CHANNELS, DoubleConv, Down, Up, xavier_init


def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """Continuous 6-D rotation param -> (..., 3, 3) via Gram-Schmidt."""
    a1, a2 = x[..., 0:3], x[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


class ConvGRUCell(nn.Module):
    """Standard ConvGRU cell (3x3 kernels)."""

    def __init__(self, ch: int):
        super().__init__()
        self.zr = nn.Conv2d(2 * ch, 2 * ch, 3, padding=1)
        self.hh = nn.Conv2d(2 * ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        zr = torch.sigmoid(self.zr(torch.cat([x, h], dim=1)))
        z, r = zr.chunk(2, dim=1)
        h_tilde = torch.tanh(self.hh(torch.cat([x, r * h], dim=1)))
        return (1 - z) * h + z * h_tilde


def _sine_pos_2d(c: int, h: int, w: int, device) -> torch.Tensor:
    """(h*w, c) 2-D sine/cosine positional encoding."""
    assert c % 4 == 0
    cq = c // 4
    y = torch.arange(h, device=device).float()[:, None].expand(h, w)
    x = torch.arange(w, device=device).float()[None, :].expand(h, w)
    div = torch.exp(torch.arange(cq, device=device).float()
                    * (-math.log(1e4) / cq))
    pe = torch.cat([
        torch.sin(x[..., None] * div), torch.cos(x[..., None] * div),
        torch.sin(y[..., None] * div), torch.cos(y[..., None] * div),
    ], dim=-1)
    return pe.reshape(h * w, c)


class GatePoseNetMG(nn.Module):
    OUT_PER_QUERY = ("presence_logit", "target_logit", "corners_uv",
                     "corner_inside_logit", "center_uv", "position",
                     "log_depth", "rot6d", "visible_frac")

    def __init__(
        self,
        in_channels: int = 3,
        channels=BASE_CHANNELS,
        width_factor: float = 2.0,
        n_queries: int = 8,
        d_model: int = 128,
        n_dec_layers: int = 2,
        n_heads: int = 4,
        ego_dim: int = 7,
        temporal_highres: bool = False,
        multiscale_memory: bool = False,
        mask_stride: int = 4,
    ):
        super().__init__()
        ch = tuple(max(1, int(round(c * width_factor / 4.0))) for c in channels)
        c1, c2, c3, c4, c5 = ch
        self.channels = ch
        self.n_queries = int(n_queries)
        self.d_model = int(d_model)
        self.temporal_highres = bool(temporal_highres)
        self.multiscale_memory = bool(multiscale_memory)
        self.mask_stride = int(mask_stride)
        if self.mask_stride not in (2, 4):
            raise ValueError("mask_stride must be 2 or 4")

        # per-frame encoder + temporal core (same recipe as GatePoseNet)
        self.inc = DoubleConv(in_channels, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.down4 = Down(c4, c5)
        self.ego_mlp = nn.Sequential(
            nn.Linear(ego_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, c5))
        self.gru = ConvGRUCell(c5)

        # Optional 1/8-resolution temporal state.  This is specifically useful
        # for small/far gates that can disappear at the 1/16 bottleneck.
        if self.temporal_highres:
            self.ego_mlp4 = nn.Sequential(
                nn.Linear(ego_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, c4))
            self.gru4 = ConvGRUCell(c4)

        # Pixel decoder.  The legacy model emits masks at 1/4 resolution;
        # far-gate mode can continue one stage to 1/2 resolution.
        self.px_up1 = Up(c5, c4)
        self.px_up2 = Up(c4, c3)
        if self.mask_stride == 2:
            self.px_up3 = Up(c3, c2)
            self.mask_feat = nn.Conv2d(c2, d_model, kernel_size=1)
        else:
            self.mask_feat = nn.Conv2d(c3, d_model, kernel_size=1)

        # Query transformer decoder.  Legacy mode attends only to the 1/16
        # ConvGRU state.  Far-gate mode also attends to 1/8 temporal features.
        self.in_proj = nn.Conv2d(c5, d_model, kernel_size=1)
        if self.multiscale_memory:
            self.in_proj4 = nn.Conv2d(c4, d_model, kernel_size=1)
            self.level_embed = nn.Parameter(torch.zeros(2, d_model))
        self.queries = nn.Embedding(self.n_queries, d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=0.0, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_dec_layers)

        # per-query heads
        n_out = 1 + 1 + 8 + 4 + 2 + 3 + 1 + 6 + 1
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, n_out))
        self.mask_embed = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model))
        xavier_init(self)

    # ------------------------------------------------------------------ #
    def _encode(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x2, x3, x4, x5

    def _decode(self, h5, h4, x2, x3, x4):
        B, C, Hs, Ws = h5.shape
        tok5 = self.in_proj(h5).flatten(2).transpose(1, 2)
        tok5 = tok5 + _sine_pos_2d(self.d_model, Hs, Ws, h5.device)[None]
        if self.multiscale_memory:
            # Prefer the temporal 1/8 state when enabled; otherwise use the
            # current-frame 1/8 encoder feature map.
            fine = h4 if h4 is not None else x4
            _, _, H4, W4 = fine.shape
            tok4 = self.in_proj4(fine).flatten(2).transpose(1, 2)
            tok4 = tok4 + _sine_pos_2d(self.d_model, H4, W4, fine.device)[None]
            tok5 = tok5 + self.level_embed[0][None, None]
            tok4 = tok4 + self.level_embed[1][None, None]
            tokens = torch.cat([tok5, tok4], dim=1)
        else:
            tokens = tok5
        q = self.queries.weight[None].expand(B, -1, -1)       # (B, Q, D)
        q = self.decoder(q, tokens)                           # (B, Q, D)

        v = self.head(q)                                      # (B, Q, n_out)
        i = 0
        presence_logit = v[..., i]; i += 1
        target_logit = v[..., i]; i += 1
        corners_uv = v[..., i:i + 8].reshape(B, -1, 4, 2); i += 8
        corner_inside_logit = v[..., i:i + 4]; i += 4
        center_uv = v[..., i:i + 2]; i += 2
        position = v[..., i:i + 3]; i += 3
        log_depth = v[..., i]; i += 1
        rot6d = v[..., i:i + 6]; i += 6
        visible_frac = torch.sigmoid(v[..., i]); i += 1

        # MaskFormer-style instance masks.  In far-gate mode the temporal
        # 1/8 skip is decoded to 1/2 resolution, retaining thin distant gates.
        skip4 = h4 if h4 is not None else x4
        px = self.px_up2(self.px_up1(h5, skip4), x3)
        if self.mask_stride == 2:
            px = self.px_up3(px, x2)
        px = self.mask_feat(px)
        me = self.mask_embed(q)                                   # (B,Q,D)
        mask_logit = torch.einsum("bqd,bdhw->bqhw", me, px)

        return dict(
            presence_logit=presence_logit, target_logit=target_logit,
            corners_uv=corners_uv, corner_inside_logit=corner_inside_logit,
            center_uv=center_uv, position=position, log_depth=log_depth,
            rot6d=rot6d, R=rot6d_to_matrix(rot6d),
            visible_frac=visible_frac, mask_logit=mask_logit)

    def init_hidden(self, batch, hw, device=None):
        c4, c5 = self.channels[-2], self.channels[-1]
        h5 = torch.zeros(batch, c5, hw[0] // 16, hw[1] // 16, device=device)
        if not self.temporal_highres:
            return h5
        h4 = torch.zeros(batch, c4, hw[0] // 8, hw[1] // 8, device=device)
        return (h5, h4)

    def step(self, image, ego, h=None):
        x2, x3, x4, x5 = self._encode(image)
        x5 = x5 + self.ego_mlp(ego)[:, :, None, None]
        if self.temporal_highres:
            if h is None:
                h5_prev, h4_prev = torch.zeros_like(x5), torch.zeros_like(x4)
            else:
                h5_prev, h4_prev = h
            h5 = self.gru(x5, h5_prev)
            x4e = x4 + self.ego_mlp4(ego)[:, :, None, None]
            h4 = self.gru4(x4e, h4_prev)
            hidden = (h5, h4)
        else:
            h5_prev = torch.zeros_like(x5) if h is None else h
            h5 = self.gru(x5, h5_prev)
            h4 = None
            hidden = h5
        return self._decode(h5, h4, x2, x3, x4), hidden

    def forward(self, images, ego):
        B, T = images.shape[:2]
        h = None
        outs = []
        for t in range(T):
            out, h = self.step(images[:, t], ego[:, t], h)
            outs.append(out)
        return {k: torch.stack([o[k] for o in outs], dim=1) for k in outs[0]}

    @torch.no_grad()
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def build_gateposenet_mg(cfg: dict) -> GatePoseNetMG:
    return GatePoseNetMG(
        in_channels=int(cfg.get("in_channels", 3)),
        channels=tuple(cfg.get("base_channels", BASE_CHANNELS)),
        width_factor=float(cfg.get("width_factor", 2.0)),
        n_queries=int(cfg.get("n_queries", 8)),
        d_model=int(cfg.get("d_model", 128)),
        n_dec_layers=int(cfg.get("n_dec_layers", 2)),
        n_heads=int(cfg.get("n_heads", 4)),
        temporal_highres=bool(cfg.get("temporal_highres", False)),
        multiscale_memory=bool(cfg.get("multiscale_memory", False)),
        mask_stride=int(cfg.get("mask_stride", 4)),
    )


# GatePoseNet-MG is THE default model of this package (multi-gate: per-gate
# masks + keypoints + pose + target scoring). The legacy single-target model
# lives in gateposenet.model_single for existing checkpoints/baselines.
GatePoseNet = GatePoseNetMG
build_gateposenet = build_gateposenet_mg
