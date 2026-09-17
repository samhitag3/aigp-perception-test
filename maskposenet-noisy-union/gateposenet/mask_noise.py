"""Training-only corruption for union-mask MaskPoseNet.

The goal is to make the model robust to realistic upstream segmentation errors
without contaminating validation/test ground truth.  Corruption is applied to
an input copy of the canonical instance mask; the original instance IDs remain
untouched and continue to supervise per-gate masks, keypoints and pose.

Noise is sampled once per temporal window and then varied mildly frame-to-frame.
This creates correlated defects that the ConvGRU can learn to bridge instead of
seeing only independent pixel noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


def _odd(v: int) -> int:
    v = max(1, int(v))
    return v if v % 2 else v + 1


def _clip01(v: float) -> float:
    return float(np.clip(v, 0.0, 1.0))


@dataclass
class _GateStyle:
    shift_xy: tuple[float, float]
    morph_sign: int
    bite_side: int
    frag_bias: float


@dataclass
class _PersistentArtifact:
    kind: str
    cx: float
    cy: float
    w: float
    h: float
    angle_deg: float
    thickness_frac: float
    missing_side: int
    drift_x: float
    drift_y: float


@dataclass
class WindowMaskNoise:
    """Stateful, temporally correlated mask corruption for one training window."""

    cfg: dict[str, Any]
    seed: int
    rng: np.random.Generator = field(init=False)
    severity: float = field(init=False)
    clean_window: bool = field(init=False)
    gate_styles: dict[int, _GateStyle] = field(default_factory=dict, init=False)
    artifacts: list[_PersistentArtifact] = field(default_factory=list, init=False)
    _artifact_shape: tuple[int, int] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(int(self.seed))
        self.clean_window = bool(self.rng.random() < float(self.cfg.get("clean_window_prob", 0.18)))
        lo = float(self.cfg.get("severity_min", 0.25))
        hi = float(self.cfg.get("severity_max", 0.95))
        self.severity = float(self.rng.uniform(min(lo, hi), max(lo, hi)))

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.get("enabled", True)) and not self.clean_window

    def _style_for(self, mask_id: int) -> _GateStyle:
        if mask_id not in self.gate_styles:
            max_shift = float(self.cfg.get("max_shift_px", 3.0)) * self.severity
            self.gate_styles[mask_id] = _GateStyle(
                shift_xy=(
                    float(self.rng.uniform(-max_shift, max_shift)),
                    float(self.rng.uniform(-max_shift, max_shift)),
                ),
                morph_sign=int(self.rng.choice([-1, 0, 1], p=[0.40, 0.20, 0.40])),
                bite_side=int(self.rng.integers(0, 4)),
                frag_bias=float(self.rng.uniform(0.7, 1.3)),
            )
        return self.gate_styles[mask_id]

    def _init_artifacts(self, shape: tuple[int, int]) -> None:
        if self._artifact_shape is not None:
            return
        self._artifact_shape = shape
        max_art = int(self.cfg.get("max_persistent_false_artifacts", 2))
        prob = float(self.cfg.get("persistent_false_artifact_prob", 0.45)) * self.severity
        n = 0
        for _ in range(max_art):
            if self.rng.random() < prob:
                n += 1
        for _ in range(n):
            self.artifacts.append(_PersistentArtifact(
                kind=str(self.rng.choice(["partial_ring", "polygon"], p=[0.72, 0.28])),
                cx=float(self.rng.uniform(0.08, 0.92)),
                cy=float(self.rng.uniform(0.08, 0.92)),
                w=float(self.rng.uniform(0.05, 0.22)),
                h=float(self.rng.uniform(0.05, 0.24)),
                angle_deg=float(self.rng.uniform(-35.0, 35.0)),
                thickness_frac=float(self.rng.uniform(0.05, 0.14)),
                missing_side=int(self.rng.integers(0, 4)),
                drift_x=float(self.rng.uniform(-0.0025, 0.0025)),
                drift_y=float(self.rng.uniform(-0.0025, 0.0025)),
            ))

    def _shift(self, bm: np.ndarray, dx: float, dy: float) -> np.ndarray:
        if abs(dx) < 0.25 and abs(dy) < 0.25:
            return bm
        h, w = bm.shape
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(bm, M, (w, h), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def _rounded_edges(self, bm: np.ndarray, local_sev: float) -> np.ndarray:
        if self.rng.random() >= float(self.cfg.get("rounded_edge_prob", 0.65)) * local_sev:
            return bm
        ys, xs = np.nonzero(bm)
        if len(xs) < 12:
            return bm
        span = max(1, min(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)))
        kmax = int(self.cfg.get("rounded_kernel_max", 7))
        k = min(kmax, max(3, int(round(span * 0.025 * (0.6 + local_sev)))))
        k = _odd(k)
        # Blur + threshold softens square mask corners; a light elliptical closing
        # reconnects thin ring sections without turning them into filled boxes.
        blur = cv2.GaussianBlur(bm, (k, k), sigmaX=0)
        out = (blur >= 105).astype(np.uint8) * 255
        ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.morphologyEx(out, cv2.MORPH_CLOSE, ek, iterations=1)

    def _morph_jitter(self, bm: np.ndarray, style: _GateStyle, local_sev: float) -> np.ndarray:
        if style.morph_sign == 0 or self.rng.random() >= float(self.cfg.get("morph_prob", 0.55)) * local_sev:
            return bm
        ys, xs = np.nonzero(bm)
        if len(xs) < 10:
            return bm
        span = max(1, min(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)))
        k = 3 if span < 100 else (5 if local_sev > 0.7 else 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        op = cv2.erode if style.morph_sign < 0 else cv2.dilate
        out = op(bm, kernel, iterations=1)
        # Don't accidentally erase a tiny/far gate unless the explicit dropout
        # augmentation selected that behavior.
        if cv2.countNonZero(out) < max(4, int(0.18 * cv2.countNonZero(bm))):
            return bm
        return out

    def _remove_chunks(self, bm: np.ndarray, style: _GateStyle, local_sev: float) -> np.ndarray:
        p = float(self.cfg.get("incomplete_prob", 0.72)) * local_sev * style.frag_bias
        if self.rng.random() >= min(p, 0.95):
            return bm
        ys, xs = np.nonzero(bm)
        if len(xs) < 8:
            return bm
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        bw, bh = max(2, x1 - x0 + 1), max(2, y1 - y0 + 1)
        out = bm.copy()
        max_chunks = int(self.cfg.get("max_missing_chunks", 3))
        n = int(self.rng.integers(1, max_chunks + 1))
        for ci in range(n):
            # Bias one chunk toward the same side across the window; extra chunks
            # vary frame to frame. This resembles a detector repeatedly missing a bar.
            side = style.bite_side if ci == 0 else int(self.rng.integers(0, 4))
            frac_w = float(self.rng.uniform(0.16, 0.45)) * (0.65 + 0.55 * local_sev)
            frac_h = float(self.rng.uniform(0.16, 0.45)) * (0.65 + 0.55 * local_sev)
            rw, rh = max(2, int(bw * frac_w)), max(2, int(bh * frac_h))
            if side == 0:  # top
                cx = int(self.rng.integers(x0, x1 + 1)); cy = y0 + int(self.rng.uniform(0, 0.24) * bh)
            elif side == 1:  # right
                cx = x1 - int(self.rng.uniform(0, 0.24) * bw); cy = int(self.rng.integers(y0, y1 + 1))
            elif side == 2:  # bottom
                cx = int(self.rng.integers(x0, x1 + 1)); cy = y1 - int(self.rng.uniform(0, 0.24) * bh)
            else:  # left
                cx = x0 + int(self.rng.uniform(0, 0.24) * bw); cy = int(self.rng.integers(y0, y1 + 1))
            if self.rng.random() < 0.55:
                cv2.ellipse(out, (cx, cy), (rw // 2, rh // 2), 0, 0, 360, 0, -1)
            else:
                cv2.rectangle(out, (cx - rw // 2, cy - rh // 2),
                              (cx + rw // 2, cy + rh // 2), 0, -1)
        # Preserve some signal except for explicit gate-dropout.
        if cv2.countNonZero(out) < max(3, int(0.10 * len(xs))):
            return bm
        return out

    def _draw_partial_ring(self, canvas: np.ndarray, art: _PersistentArtifact,
                           frame_pos: int) -> None:
        h, w = canvas.shape
        cx = (art.cx + art.drift_x * frame_pos) * w
        cy = (art.cy + art.drift_y * frame_pos) * h
        rw = max(5.0, art.w * w)
        rh = max(5.0, art.h * h)
        theta = np.deg2rad(art.angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        pts = np.array([[-rw/2,-rh/2],[rw/2,-rh/2],[rw/2,rh/2],[-rw/2,rh/2]], dtype=np.float32)
        R = np.array([[c,-s],[s,c]], dtype=np.float32)
        pts = pts @ R.T + np.array([cx,cy], dtype=np.float32)
        pts = np.round(pts).astype(np.int32)
        thickness = max(1, int(min(rw, rh) * art.thickness_frac))
        for side in range(4):
            if side == art.missing_side:
                continue
            p0 = tuple(pts[side]); p1 = tuple(pts[(side + 1) % 4])
            cv2.line(canvas, p0, p1, 255, thickness=thickness, lineType=cv2.LINE_AA)

    def _draw_polygon(self, canvas: np.ndarray, art: _PersistentArtifact,
                      frame_pos: int) -> None:
        h, w = canvas.shape
        cx = (art.cx + art.drift_x * frame_pos) * w
        cy = (art.cy + art.drift_y * frame_pos) * h
        rw = max(4.0, art.w * w)
        rh = max(4.0, art.h * h)
        n = int(self.rng.integers(3, 7))
        ang = np.sort(self.rng.uniform(0, 2*np.pi, size=n))
        rad = self.rng.uniform(0.45, 1.0, size=n)
        pts = np.stack([cx + np.cos(ang)*rw*0.5*rad,
                        cy + np.sin(ang)*rh*0.5*rad], axis=1)
        pts = np.round(pts).astype(np.int32)
        if self.rng.random() < 0.55:
            cv2.fillPoly(canvas, [pts], 255)
        else:
            cv2.polylines(canvas, [pts], True, 255,
                          thickness=max(1, int(min(rw, rh)*0.08)), lineType=cv2.LINE_AA)

    def _add_false_artifacts(self, union: np.ndarray, frame_pos: int,
                             local_sev: float) -> np.ndarray:
        out = union.copy()
        for art in self.artifacts:
            # Persistent artifacts can flicker but usually survive several frames.
            if self.rng.random() < 0.12 * (1.0 - local_sev):
                continue
            if art.kind == "partial_ring":
                self._draw_partial_ring(out, art, frame_pos)
            else:
                self._draw_polygon(out, art, frame_pos)

        # Frame-local speckles/blobs emulate small false components.
        if self.rng.random() < float(self.cfg.get("small_false_blob_prob", 0.28)) * local_sev:
            h, w = out.shape
            n = int(self.rng.integers(1, int(self.cfg.get("max_small_false_blobs", 4)) + 1))
            for _ in range(n):
                cx, cy = int(self.rng.integers(0, w)), int(self.rng.integers(0, h))
                rx = int(self.rng.integers(2, max(3, int(0.018*w))))
                ry = int(self.rng.integers(2, max(3, int(0.018*h))))
                cv2.ellipse(out, (cx,cy), (rx,ry), float(self.rng.uniform(0,180)), 0, 360, 255, -1)
        return out

    def apply(self, mask_ids: np.ndarray, frame_pos: int = 0) -> np.ndarray:
        """Return a noisy binary union mask (uint8 0/255) for model input."""
        clean_union = (mask_ids > 0).astype(np.uint8) * 255
        if not self.enabled:
            return clean_union

        self._init_artifacts(clean_union.shape)
        # Corruption strength varies smoothly-ish inside a shared window profile.
        jitter = float(self.rng.uniform(0.72, 1.18))
        local_sev = _clip01(self.severity * jitter)

        # Rare complete detector dropout for this frame. ConvGRU should bridge it.
        if self.rng.random() < float(self.cfg.get("full_frame_dropout_prob", 0.025)) * local_sev:
            union = np.zeros_like(clean_union)
            return self._add_false_artifacts(union, frame_pos, local_sev)

        union = np.zeros_like(clean_union)
        ids = [int(x) for x in np.unique(mask_ids) if int(x) > 0]
        gate_drop_p = float(self.cfg.get("gate_dropout_prob", 0.07)) * local_sev

        for mid in ids:
            bm = (mask_ids == mid).astype(np.uint8) * 255
            if cv2.countNonZero(bm) == 0:
                continue
            # Whole-instance miss: deliberately independent per frame, but rare.
            if self.rng.random() < gate_drop_p:
                continue
            style = self._style_for(mid)
            # Persistent directional bias with mild frame jitter.
            dx = style.shift_xy[0] + float(self.rng.normal(0, 0.55 * local_sev))
            dy = style.shift_xy[1] + float(self.rng.normal(0, 0.55 * local_sev))
            bm = self._shift(bm, dx, dy)
            bm = self._rounded_edges(bm, local_sev)
            bm = self._morph_jitter(bm, style, local_sev)
            bm = self._remove_chunks(bm, style, local_sev)
            union = cv2.bitwise_or(union, bm)

        union = self._add_false_artifacts(union, frame_pos, local_sev)
        return (union > 0).astype(np.uint8) * 255


def mask_noise_enabled(cfg: dict[str, Any] | None) -> bool:
    return bool(cfg and cfg.get("enabled", True))
