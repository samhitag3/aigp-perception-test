"""GatePoseNetSingle — LEGACY single-target temporal model.

Superseded by the multi-gate default (gateposenet.model / GatePoseNet-MG);
kept for existing checkpoints (runs/gateposenet_traj/*) and baselines.

Design (see README.md for the full rationale):

* **Per-frame encoder**: the GateNet double-conv/Down blocks (MonoRace-style
  U-Net encoder, width-scaled) — reused from :mod:`gatenet.model` so the two
  models share the proven lightweight backbone.
* **Temporal core**: a single ConvGRU cell on the 1/16-resolution bottleneck.
  The recurrent state carries the gate through partial views and the blind
  fly-through moment (SkyDreamer keeps a GRU latent for exactly this reason).
* **Ego-motion conditioning**: [v_cam, omega_cam] (from the trajectory GT in
  sim; the flight controller's EKF at deployment) is embedded by a small MLP
  and ADDED to the bottleneck features before the GRU — without ego-motion the
  network cannot know how the camera moved while blind. Trained with modality
  dropout so it degrades gracefully when the estimate is missing.
* **Heads** (from pooled GRU state):
  - corners_uv (4x2, image-normalized) + per-corner inside-frame logits — the
    MonoRace path: corners -> PnP when calibration is available, with partial
    corner sets usable;
  - center_uv (2);
  - position_cam (3, meters) + log_depth (1) — the direct metric path (gate
    size is known, so scale is observable without calibration);
  - rot6d (6) -> R_cam_gate via Gram-Schmidt (Zhou et al. CVPR'19 continuous
    rotation representation); normal_cam = -R[:, 2];
  - visibility logit + visible_frac;
  - k_scale (2) — auxiliary self-calibration decode (fx, fy relative to
    nominal), the intrinsics analog of SkyDreamer's privileged decoding.
* **Aux segmentation decoder** at 1/4 resolution keeps the encoder honest
  about WHERE the gate is (deep supervision of localization).

The model exposes ``forward`` over (B, T, ...) windows for training and
``step`` (single frame + hidden state) for 30 Hz deployment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatenet.model import BASE_CHANNELS, DoubleConv, Down, Up, xavier_init
from gateposenet.model import ConvGRUCell, rot6d_to_matrix


class GatePoseNetSingle(nn.Module):
    OUT_KEYS = ("corners_uv", "corner_inside_logit", "center_uv", "position",
                "log_depth", "rot6d", "visible_logit", "visible_frac",
                "k_scale")

    def __init__(
        self,
        in_channels: int = 3,
        channels=BASE_CHANNELS,
        width_factor: float = 2.0,
        head_hidden: int = 256,
        ego_dim: int = 7,   # [v_cam(3), omega_cam(3), dt] — dt = rate-awareness
        max_depth_m: float = 25.0,
    ):
        super().__init__()
        ch = tuple(max(1, int(round(c * width_factor / 4.0))) for c in channels)
        c1, c2, c3, c4, c5 = ch
        self.channels = ch
        self.max_depth_m = float(max_depth_m)

        # per-frame encoder (GateNet blocks)
        self.inc = DoubleConv(in_channels, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.down4 = Down(c4, c5)

        # ego-motion embedding -> bottleneck bias
        self.ego_mlp = nn.Sequential(
            nn.Linear(ego_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, c5),
        )

        # temporal core
        self.gru = ConvGRUCell(c5)

        # aux seg decoder to 1/4 resolution (uses c4/c3 skips)
        self.seg_up1 = Up(c5, c4)
        self.seg_up2 = Up(c4, c3)
        self.seg_out = nn.Conv2d(c3, 1, kernel_size=1)

        # pose head from pooled GRU state (avg + max pool)
        n_out = 8 + 4 + 2 + 3 + 1 + 6 + 1 + 1 + 2
        self.head = nn.Sequential(
            nn.Linear(2 * c5, head_hidden), nn.ReLU(inplace=True),
            nn.Linear(head_hidden, head_hidden), nn.ReLU(inplace=True),
            nn.Linear(head_hidden, n_out),
        )
        xavier_init(self)

    # ------------------------------------------------------------------ #
    def _encode(self, x: torch.Tensor):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x3, x4, x5

    def _decode_outputs(self, h: torch.Tensor, x3, x4):
        seg = self.seg_out(self.seg_up2(self.seg_up1(h, x4), x3))
        pooled = torch.cat([
            F.adaptive_avg_pool2d(h, 1).flatten(1),
            F.adaptive_max_pool2d(h, 1).flatten(1),
        ], dim=1)
        v = self.head(pooled)
        i = 0
        corners_uv = v[:, i:i + 8].reshape(-1, 4, 2); i += 8
        corner_inside_logit = v[:, i:i + 4]; i += 4
        center_uv = v[:, i:i + 2]; i += 2
        position = v[:, i:i + 3]; i += 3
        log_depth = v[:, i:i + 1].squeeze(-1); i += 1
        rot6d = v[:, i:i + 6]; i += 6
        visible_logit = v[:, i:i + 1].squeeze(-1); i += 1
        visible_frac = torch.sigmoid(v[:, i:i + 1]).squeeze(-1); i += 1
        k_scale = 1.0 + 0.25 * torch.tanh(v[:, i:i + 2]); i += 2
        return dict(
            seg_logit=seg,
            corners_uv=corners_uv,
            corner_inside_logit=corner_inside_logit,
            center_uv=center_uv,
            position=position,
            log_depth=log_depth,
            rot6d=rot6d,
            R=rot6d_to_matrix(rot6d),
            visible_logit=visible_logit,
            visible_frac=visible_frac,
            k_scale=k_scale,
        )

    def init_hidden(self, batch: int, hw: tuple[int, int],
                    device=None) -> torch.Tensor:
        c5 = self.channels[-1]
        return torch.zeros(batch, c5, hw[0] // 16, hw[1] // 16, device=device)

    def step(self, image: torch.Tensor, ego: torch.Tensor,
             h: torch.Tensor | None = None):
        """Single-frame deployment step. image (B,3,H,W), ego (B,6)."""
        x3, x4, x5 = self._encode(image)
        x5 = x5 + self.ego_mlp(ego)[:, :, None, None]
        if h is None:
            h = torch.zeros_like(x5)
        h = self.gru(x5, h)
        out = self._decode_outputs(h, x3, x4)
        return out, h

    def forward(self, images: torch.Tensor, ego: torch.Tensor):
        """images (B,T,3,H,W), ego (B,T,6) -> dict of (B,T,...) outputs."""
        B, T = images.shape[:2]
        h = None
        outs: list[dict] = []
        for t in range(T):
            out, h = self.step(images[:, t], ego[:, t], h)
            outs.append(out)
        stacked = {k: torch.stack([o[k] for o in outs], dim=1)
                   for k in outs[0]}
        return stacked

    @torch.no_grad()
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_gateposenet_single(cfg: dict) -> GatePoseNetSingle:
    return GatePoseNetSingle(
        in_channels=int(cfg.get("in_channels", 3)),
        channels=tuple(cfg.get("base_channels", BASE_CHANNELS)),
        width_factor=float(cfg.get("width_factor", 2.0)),
        head_hidden=int(cfg.get("head_hidden", 256)),
        max_depth_m=float(cfg.get("max_depth_m", 25.0)),
    )
