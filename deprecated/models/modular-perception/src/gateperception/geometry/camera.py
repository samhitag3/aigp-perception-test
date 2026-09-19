from __future__ import annotations
import math
import numpy as np


def camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def fov_from_intrinsics(width_px: int, height_px: int, fx: float, fy: float) -> tuple[float, float]:
    hfov = math.degrees(2.0 * math.atan(width_px / (2.0 * fx)))
    vfov = math.degrees(2.0 * math.atan(height_px / (2.0 * fy)))
    return hfov, vfov
