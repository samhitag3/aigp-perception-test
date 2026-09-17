"""Gate orientation/pose extraction from a GateNet segmentation mask.

Pipeline:  mask -> outer quad corners -> PnP (planar square) -> 6-DoF pose.

From the recovered rotation we expose the gate's **orientation** (the plane
normal in the camera frame + roll/pitch/yaw) and **distance**, which is what the
original GateNet (open-airlab) and the MonoRace state estimator ultimately need.

PnP rotation is scale-invariant, so orientation is correct for any assumed gate
size; only the absolute distance depends on the true gate side length.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def clean_mask(mask: np.ndarray, min_area_frac: float = 0.003,
               open_ksize: int = 3) -> np.ndarray:
    """Denoise a predicted mask: morphological open + drop tiny components.

    Removes the speckle noise GateNet produces in the background while keeping
    real gate(s). Returns a uint8 {0,1} mask.
    """
    m = (mask > 0).astype(np.uint8)
    if open_ksize >= 2:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m
    min_area = max(40, int(min_area_frac * m.shape[0] * m.shape[1]))
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lbl == i] = 1
    return out


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL (image convention, y-down)."""
    pts = np.asarray(pts, np.float32)
    s = pts.sum(1)
    d = pts[:, 1] - pts[:, 0]          # y - x
    return np.array([pts[np.argmin(s)],  # TL  (small x+y)
                     pts[np.argmin(d)],  # TR  (small y-x)
                     pts[np.argmax(s)],  # BR  (large x+y)
                     pts[np.argmax(d)]], # BL  (large y-x)
                    np.float32)


def quad_from_mask(mask: np.ndarray):
    """Largest-contour outer quadrilateral (perspective), ordered TL/TR/BR/BL.

    Returns (4,2) float corners or None if no gate found.
    """
    m = clean_mask(mask)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 50:
        return None
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    quad = None
    for k in np.linspace(0.01, 0.12, 12):
        approx = cv2.approxPolyDP(hull, k * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(-1, 2).astype(np.float32)
            break
    if quad is None:                              # fallback: min-area rect
        quad = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)
    # reject degenerate / image-spanning detections (over-prediction artifacts)
    H, W = mask.shape
    area = cv2.contourArea(quad.astype(np.float32))
    if area > 0.6 * H * W or area < 80:
        return None
    border = sum(1 for x, y in quad if x < 2 or y < 2 or x > W - 3 or y > H - 3)
    if border >= 3:
        return None
    return order_quad(quad)


@dataclass
class GatePose:
    corners: np.ndarray      # (4,2) image corners TL/TR/BR/BL
    rvec: np.ndarray
    tvec: np.ndarray
    R: np.ndarray            # gate->camera rotation
    normal_cam: np.ndarray   # gate plane normal in camera frame (unit)
    rpy_deg: np.ndarray      # roll, pitch, yaw (deg)
    distance: float          # ||tvec|| in the chosen gate-size units
    reproj_err: float


def _rpy_from_R(R: np.ndarray) -> np.ndarray:
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1]); pitch = np.arctan2(-R[2, 0], sy); yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def pose_from_corners(corners: np.ndarray, K: np.ndarray, side: float = 1.0):
    """PnP for a planar square gate. corners ordered TL/TR/BR/BL."""
    h = side / 2.0
    obj = np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]], np.float64)
    img = np.asarray(corners, np.float64)
    try:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    except cv2.error:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    normal = R @ np.array([0.0, 0.0, 1.0])
    if normal[2] > 0:            # point the normal toward the camera (-z)
        normal = -normal
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    reproj = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)))
    return GatePose(corners=np.asarray(corners, np.float32), rvec=rvec, tvec=tvec,
                    R=R, normal_cam=normal / (np.linalg.norm(normal) + 1e-9),
                    rpy_deg=_rpy_from_R(R), distance=float(np.linalg.norm(tvec)),
                    reproj_err=reproj)


def gate_pose_from_mask(mask: np.ndarray, K: np.ndarray, side: float = 1.0):
    """Convenience: mask -> quad -> pose. Returns (GatePose|None, quad|None)."""
    quad = quad_from_mask(mask)
    if quad is None:
        return None, None
    return pose_from_corners(quad, K, side=side), quad


def normal_angle_error_deg(n_est: np.ndarray, n_gt: np.ndarray) -> float:
    """Angle between two gate normals (deg), sign-invariant."""
    a = n_est / (np.linalg.norm(n_est) + 1e-9)
    b = np.asarray(n_gt, float); b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(abs(float(a @ b)), 0, 1))))


def draw_pose(img, pose: GatePose, K: np.ndarray, side: float = 1.0):
    """Draw the gate quad + projected 3D axes onto a BGR image."""
    out = img.copy()
    q = pose.corners.astype(int)
    cv2.polylines(out, [q], True, (0, 255, 0), 2)
    for (x, y) in q:
        cv2.circle(out, (int(x), int(y)), 4, (0, 255, 255), -1)
    cv2.drawFrameAxes(out, K, None, pose.rvec, pose.tvec, side * 0.5, 2)
    return out
