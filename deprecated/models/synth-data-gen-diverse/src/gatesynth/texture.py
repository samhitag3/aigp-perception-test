from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np

OG_GATE_DICT = {
    "outer": [[117, 117], [906, 117], [906, 906], [117, 906]],
    "inner": [[292, 292], [731, 292], [731, 731], [292, 731]],
}


def load_gate_skin(path: str | Path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read gate skin: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    alpha = np.zeros((h, w), np.uint8)
    outer = np.asarray(OG_GATE_DICT["outer"], np.int32)
    inner = np.asarray(OG_GATE_DICT["inner"], np.int32)
    cv2.fillConvexPoly(alpha, outer, 255)
    cv2.fillConvexPoly(alpha, inner, 0)
    rgba = np.dstack([rgb, alpha])
    return rgba


def warp_gate_rgba(rgba: np.ndarray, dst_outer_uv: np.ndarray, out_size: tuple[int, int]):
    src = np.asarray(OG_GATE_DICT["outer"], np.float32)
    dst = np.asarray(dst_outer_uv, np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    w, h = out_size
    warped = cv2.warpPerspective(rgba, H, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return warped
