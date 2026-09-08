from __future__ import annotations
from dataclasses import dataclass
import math
import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxTransform:
    src_w: int
    src_h: int
    dst_w: int
    dst_h: int
    scale: float
    pad_x: int
    pad_y: int


def fov_from_intrinsics(width: int, height: int, fx: float, fy: float) -> tuple[float, float]:
    hfov = math.degrees(2.0 * math.atan(width / (2.0 * fx)))
    vfov = math.degrees(2.0 * math.atan(height / (2.0 * fy)))
    return hfov, vfov


def letterbox_params(src_w: int, src_h: int, dst_w: int, dst_h: int) -> LetterboxTransform:
    scale = min(dst_w / src_w, dst_h / src_h)
    new_w, new_h = int(round(src_w * scale)), int(round(src_h * scale))
    pad_x = (dst_w - new_w) // 2
    pad_y = (dst_h - new_h) // 2
    return LetterboxTransform(src_w, src_h, dst_w, dst_h, scale, pad_x, pad_y)


def letterbox_image(image: np.ndarray, dst_w: int, dst_h: int, interpolation=cv2.INTER_AREA):
    h, w = image.shape[:2]
    tr = letterbox_params(w, h, dst_w, dst_h)
    nw, nh = int(round(w * tr.scale)), int(round(h * tr.scale))
    resized = cv2.resize(image, (nw, nh), interpolation=interpolation)
    if image.ndim == 3:
        canvas = np.zeros((dst_h, dst_w, image.shape[2]), dtype=image.dtype)
    else:
        canvas = np.zeros((dst_h, dst_w), dtype=image.dtype)
    canvas[tr.pad_y:tr.pad_y+nh, tr.pad_x:tr.pad_x+nw] = resized
    return canvas, tr


def transform_K(K: np.ndarray, tr: LetterboxTransform) -> np.ndarray:
    out = K.astype(np.float32).copy()
    out[0, 0] *= tr.scale
    out[1, 1] *= tr.scale
    out[0, 2] = out[0, 2] * tr.scale + tr.pad_x
    out[1, 2] = out[1, 2] * tr.scale + tr.pad_y
    return out


def transform_points(points_xy: np.ndarray, tr: LetterboxTransform) -> np.ndarray:
    p = points_xy.astype(np.float32).copy()
    p[..., 0] = p[..., 0] * tr.scale + tr.pad_x
    p[..., 1] = p[..., 1] * tr.scale + tr.pad_y
    return p


def inverse_points(points_xy: np.ndarray, tr: LetterboxTransform) -> np.ndarray:
    p = points_xy.astype(np.float32).copy()
    p[..., 0] = (p[..., 0] - tr.pad_x) / tr.scale
    p[..., 1] = (p[..., 1] - tr.pad_y) / tr.scale
    return p
