"""Synthetic gate-image generator (MonoRace recipe).

Each sample composites one (or more) gate face onto a background:
  * random scale, in-plane rotation, and perspective warp of the gate,
  * the gate is alpha-composited onto a random background crop,
  * independent HSV jitter on gate and background,
  * additive Gaussian noise on the final image.
The binary mask is the warped gate alpha (the gate *frame* pixels).

INPUTS
------
gates_dir:       PNG gate-face templates. RGBA preferred (alpha = gate frame).
                 If a gate has no alpha channel, near-black or near-white pixels
                 are treated as background depending on ``alpha_from``.
backgrounds_dir: arbitrary background images (warehouses, halls, etc.).

This can be used as an on-the-fly ``torch`` Dataset (``SyntheticGateDataset``)
or dumped to disk with ``scripts/make_synthetic.py``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def _list(d: Path):
    return sorted(p for p in Path(d).iterdir() if p.suffix.lower() in IMG_EXTS)


def _load_rgba(path: Path, alpha_from: str = "alpha"):
    """Return (rgb uint8 HxWx3, alpha float HxW in [0,1])."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"cannot read {path}")
    if raw.ndim == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    if raw.shape[2] == 4:
        rgb = cv2.cvtColor(raw[..., :3], cv2.COLOR_BGR2RGB)
        alpha = raw[..., 3].astype(np.float32) / 255.0
    else:
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        if alpha_from == "white":
            alpha = (gray < 230).astype(np.float32)
        else:  # "black" / default: treat near-black as background
            alpha = (gray > 25).astype(np.float32)
    return rgb, alpha


def _hsv_jitter(img, dh=12.0, ds=0.4, dv=0.4, rng=np.random):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-dh, dh)) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + rng.uniform(-ds, ds)), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * (1 + rng.uniform(-dv, dv)), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _perspective_pts(w, h, jitter, rng):
    """Random destination quad for a perspective warp (jitter as frac of size)."""
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dx, dy = jitter * w, jitter * h
    offset = rng.uniform(-1, 1, size=(4, 2)) * np.array([dx, dy])
    dst = (src + offset).astype(np.float32)
    return src, dst


def compose_sample(gate_rgb, gate_alpha, bg, size, rng=np.random,
                   scale=(0.25, 0.95), perspective=0.18,
                   hsv_gate=(12.0, 0.4, 0.4), hsv_bg=(12.0, 0.4, 0.4),
                   noise_std=(0.0, 20.0)):
    """Return (image uint8 HxWx3 RGB, mask float HxW {0,1})."""
    H, W = size
    bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_LINEAR)
    bg = _hsv_jitter(bg, *hsv_bg, rng=rng)

    # scale gate to a target size, keeping aspect ratio
    gh, gw = gate_rgb.shape[:2]
    target = rng.uniform(*scale) * min(H, W)
    s = target / max(gh, gw)
    nw, nh = max(2, int(gw * s)), max(2, int(gh * s))
    g = cv2.resize(gate_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    a = cv2.resize(gate_alpha, (nw, nh), interpolation=cv2.INTER_LINEAR)
    g = _hsv_jitter(g, *hsv_gate, rng=rng)

    # perspective + rotation warp on a canvas the size of the output
    canvas = np.zeros((H, W, 3), np.uint8)
    calpha = np.zeros((H, W), np.float32)
    ox = int(rng.uniform(0, max(1, W - nw)))
    oy = int(rng.uniform(0, max(1, H - nh)))
    canvas[oy:oy + nh, ox:ox + nw] = g
    calpha[oy:oy + nh, ox:ox + nw] = a

    src, dst = _perspective_pts(W, H, perspective, rng)
    M = cv2.getPerspectiveTransform(src, dst)
    canvas = cv2.warpPerspective(canvas, M, (W, H))
    calpha = cv2.warpPerspective(calpha, M, (W, H))
    angle = rng.uniform(-180, 180)
    R = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    canvas = cv2.warpAffine(canvas, R, (W, H))
    calpha = cv2.warpAffine(calpha, R, (W, H))

    alpha3 = np.clip(calpha, 0, 1)[..., None]
    out = (canvas.astype(np.float32) * alpha3 + bg.astype(np.float32) * (1 - alpha3))

    std = rng.uniform(*noise_std)
    if std > 0:
        out = out + rng.normal(0, std, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    mask = (calpha > 0.5).astype(np.float32)
    return out, mask


class SyntheticGateDataset(Dataset):
    """On-the-fly synthetic gate dataset."""

    def __init__(self, gates_dir, backgrounds_dir, size, length=4000,
                 augment=None, alpha_from="alpha", seed=0, **compose_kw):
        self.gates = _list(gates_dir)
        self.bgs = _list(backgrounds_dir)
        if not self.gates:
            raise RuntimeError(f"no gate templates in {gates_dir}")
        if not self.bgs:
            raise RuntimeError(f"no backgrounds in {backgrounds_dir}")
        self.size = tuple(size)
        self.length = length
        self.augment = augment
        self.alpha_from = alpha_from
        self.compose_kw = compose_kw
        self._cache = {}
        self.seed = seed

    def __len__(self):
        return self.length

    def _gate(self, i):
        if i not in self._cache:
            self._cache[i] = _load_rgba(self.gates[i], self.alpha_from)
        return self._cache[i]

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        grgb, ga = self._gate(int(rng.integers(0, len(self.gates))))
        bg = cv2.imread(str(self.bgs[int(rng.integers(0, len(self.bgs)))]),
                        cv2.IMREAD_COLOR)
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        img, mask = compose_sample(grgb, ga, bg, self.size, rng=rng,
                                   **self.compose_kw)
        if self.augment is not None:
            img, mask = self.augment(img, mask)
        img_t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(np.ascontiguousarray(mask))[None]
        return img_t, mask_t
