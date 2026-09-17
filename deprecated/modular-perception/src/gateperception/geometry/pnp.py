from __future__ import annotations
import math
from typing import Any
import cv2
import numpy as np

from gateperception.data.canonical import KEYPOINT_ORDER


def _quat_xyzw_from_R(R: np.ndarray) -> list[float]:
    # Numerically stable conversion via Rodrigues->rotation matrix elements.
    tr = float(np.trace(R))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / s; qx = 0.25 * s; qy = (R[0, 1] + R[1, 0]) / s; qz = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / s; qx = (R[0, 1] + R[1, 0]) / s; qy = 0.25 * s; qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / s; qx = (R[0, 2] + R[2, 0]) / s; qy = (R[1, 2] + R[2, 1]) / s; qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= max(np.linalg.norm(q), 1e-12)
    return q.tolist()


def solve_gate_pose(
    keypoints: dict[str, Any],
    object_points_by_name: dict[str, list[float]],
    K: np.ndarray,
    distortion: np.ndarray | None = None,
    min_confidence: float = 0.25,
    max_reprojection_error_px: float = 20.0,
) -> dict[str, Any] | None:
    obj, img, confs = [], [], []
    for name in KEYPOINT_ORDER:
        rec = keypoints.get(name)
        if not rec:
            continue
        conf = float(rec.get("confidence", 0.0))
        state = rec.get("visibility", "invalid")
        if conf < min_confidence or state == "invalid":
            continue
        xy = rec.get("xy_px")
        if xy is None:
            continue
        obj.append(object_points_by_name[name])
        img.append(xy)
        confs.append(conf)
    if len(obj) < 4:
        return None
    objp = np.asarray(obj, dtype=np.float64)
    imgp = np.asarray(img, dtype=np.float64)
    dist = np.zeros((5, 1), dtype=np.float64) if distortion is None else np.asarray(distortion, dtype=np.float64).reshape(-1, 1)

    # Gate points are coplanar, so IPPE is the appropriate planar PnP initializer.
    ok, rvecs, tvecs, reproj = cv2.solvePnPGeneric(objp, imgp, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok or len(rvecs) == 0:
        ok2, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok2:
            return None
        rvecs, tvecs = [rvec], [tvec]

    candidates = []
    for rvec, tvec in zip(rvecs, tvecs):
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        proj = proj[:, 0, :]
        err = np.linalg.norm(proj - imgp, axis=1)
        weighted = float(np.average(err, weights=np.maximum(np.asarray(confs), 1e-3)))
        z = float(np.asarray(tvec).reshape(3)[2])
        if z > 0:
            candidates.append((weighted, rvec, tvec))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    mean_err, rvec, tvec = candidates[0]
    try:
        rvec, tvec = cv2.solvePnPRefineLM(objp, imgp, K, dist, rvec, tvec)
    except cv2.error:
        pass
    R, _ = cv2.Rodrigues(rvec)
    t = np.asarray(tvec).reshape(3)
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    errs = np.linalg.norm(proj[:, 0, :] - imgp, axis=1)
    mean_err = float(np.mean(errs))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    pose_conf = float(np.clip(np.mean(confs) * math.exp(-mean_err / max(max_reprojection_error_px, 1e-6)), 0.0, 1.0))
    return {
        "T_camera_gate": T.tolist(),
        "translation_camera_m": t.tolist(),
        "quaternion_camera_xyzw": _quat_xyzw_from_R(R),
        "distance_camera_m": float(np.linalg.norm(t)),
        "depth_camera_z_m": float(t[2]),
        "mean_reprojection_error_px": mean_err,
        "max_reprojection_error_px": float(np.max(errs)),
        "num_keypoints_used": int(len(obj)),
        "confidence": pose_conf,
    }
