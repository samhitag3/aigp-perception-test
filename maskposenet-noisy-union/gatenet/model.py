"""GateNet model.

A U-Net style binary gate-segmentation network with deep supervision, exactly
following the description in the MonoRace / SkyDreamer papers:

  * Each block = double 3x3 conv -> BatchNorm -> ReLU.
  * ``inc``    : initial double-conv block.
  * ``down_k`` : max-pool -> double-conv with k output channels
                 (features stored as skip connection BEFORE pooling).
  * ``up_k``   : transposed conv + BN, ADD the corresponding encoder skip
                 (note: addition, not concatenation), then double-conv (k ch).
  * ``outc``   : 1x1 conv -> single channel (+ sigmoid at inference).
  * Channel widths are scaled by a factor ``f``.
  * Produces five output maps {y0..y4} at progressively increasing resolution
    for deep supervision; at deployment only the highest-res map (y0) is used.
  * Xavier-uniform weight initialisation.

NOTE ON CHANNEL WIDTHS
----------------------
The papers give the exact per-layer channel counts only as a figure (an image
we could not OCR), stating merely that widths "are scaled by a factor f", with
f=4 used for the 384x384 A2RL/orange model and f=2 for the 196x196 MAVLab model.
We therefore parametrise widths as ``channels[i] = base[i] * f / 4`` so that
**f=4 reproduces the nominal width** ``base`` and f=2 gives a half-width model.
Both ``base`` and ``f`` are configurable; set an explicit ``channels`` list in
the config if you know the true counts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Nominal channel widths (the f=4 model). Override via config if known exactly.
BASE_CHANNELS = (16, 32, 64, 128, 256)


def xavier_init(module: nn.Module) -> None:
    """Xavier-uniform init for all conv / transposed-conv layers (Glorot 2010)."""
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class DoubleConv(nn.Module):
    """(conv 3x3 -> BN -> ReLU) x2 — the basic GateNet block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """max-pool 2x2 then double-conv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Transposed-conv + BN + ReLU, ADD the encoder skip, then double-conv.

    Per the GateNet description the skip connection is *added* (element-wise),
    not concatenated, which keeps the network lightweight.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.conv = DoubleConv(out_ch, out_ch)

    def forward(self, x, skip):
        x = self.act(self.bn(self.up(x)))
        if x.shape[-2:] != skip.shape[-2:]:  # guard against odd input sizes
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = x + skip
        return self.conv(x)


class OutConv(nn.Module):
    """1x1 conv -> single channel. Returns logits (sigmoid applied downstream)."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class GateNet(nn.Module):
    """U-Net gate-segmentation network with deep supervision.

    Args:
        in_channels: number of input image channels (3 for RGB).
        channels: 5 encoder/decoder widths (c1..c5), c5 = bottleneck.
        deep_supervision: if True, return 5 logit maps [y0..y4] (y0 highest
            resolution). If False, return [y0] only.

    Forward returns a list of raw logit maps, highest resolution first.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels=BASE_CHANNELS,
        deep_supervision: bool = True,
    ):
        super().__init__()
        assert len(channels) == 5, "channels must have 5 entries (c1..c5)"
        c1, c2, c3, c4, c5 = channels
        self.deep_supervision = deep_supervision

        self.inc = DoubleConv(in_channels, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.down4 = Down(c4, c5)  # bottleneck

        self.up1 = Up(c5, c4)
        self.up2 = Up(c4, c3)
        self.up3 = Up(c3, c2)
        self.up4 = Up(c2, c1)

        self.outc0 = OutConv(c1)  # full resolution
        if deep_supervision:
            self.outc1 = OutConv(c2)  # 1/2
            self.outc2 = OutConv(c3)  # 1/4
            self.outc3 = OutConv(c4)  # 1/8
            self.outc4 = OutConv(c5)  # 1/16 (bottleneck)

        xavier_init(self)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # bottleneck

        d4 = self.up1(x5, x4)
        d3 = self.up2(d4, x3)
        d2 = self.up3(d3, x2)
        d1 = self.up4(d2, x1)

        y0 = self.outc0(d1)
        if not self.deep_supervision:
            return [y0]
        y1 = self.outc1(d2)
        y2 = self.outc2(d3)
        y3 = self.outc3(d4)
        y4 = self.outc4(x5)
        return [y0, y1, y2, y3, y4]

    @torch.no_grad()
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class GateNetInference(nn.Module):
    """Deployment wrapper: returns a single probability map from y0.

    Optionally resizes the output (e.g. to 64x64 as SkyDreamer feeds its policy)
    and/or thresholds to a binary mask. Suitable for ONNX/TensorRT export.
    """

    def __init__(self, net: GateNet, out_size: int | None = None,
                 threshold: float | None = None):
        super().__init__()
        self.net = net
        self.out_size = out_size
        self.threshold = threshold

    def forward(self, x):
        y0 = self.net(x)[0]
        prob = torch.sigmoid(y0)
        if self.out_size is not None:
            prob = F.interpolate(
                prob, size=(self.out_size, self.out_size),
                mode="bilinear", align_corners=False,
            )
        if self.threshold is not None:
            prob = (prob > self.threshold).float()
        return prob


def build_gatenet(cfg: dict) -> GateNet:
    """Build a GateNet from a model config dict.

    Recognised keys:
        in_channels (int, default 3)
        channels (list[int], optional): explicit 5 widths; overrides base/f.
        base_channels (list[int], default BASE_CHANNELS)
        width_factor (float, default 4): f; widths = base * f / 4.
        deep_supervision (bool, default True)
    """
    in_channels = int(cfg.get("in_channels", 3))
    deep = bool(cfg.get("deep_supervision", True))

    if cfg.get("channels"):
        channels = tuple(int(c) for c in cfg["channels"])
    else:
        base = tuple(cfg.get("base_channels", BASE_CHANNELS))
        f = float(cfg.get("width_factor", 4))
        channels = tuple(max(1, int(round(b * f / 4.0))) for b in base)

    return GateNet(in_channels=in_channels, channels=channels,
                   deep_supervision=deep)
