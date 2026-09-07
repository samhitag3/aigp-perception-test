from __future__ import annotations
import math
import numpy as np
from scipy.spatial.transform import Rotation

OUTER_W = 2.7
OUTER_H = 2.7
INNER_W = 1.5
INNER_H = 1.5
DEPTH = 0.26

KEYPOINT_ORDER = [
    "outer_tl", "outer_tr", "outer_br", "outer_bl",
    "inner_tl", "inner_tr", "inner_br", "inner_bl",
]


def gate_keypoints_local() -> dict[str, np.ndarray]:
    z = -DEPTH / 2.0
    ow, oh = OUTER_W / 2.0, OUTER_H / 2.0
    iw, ih = INNER_W / 2.0, INNER_H / 2.0
    return {
        "outer_tl": np.array([-ow, -oh, z], np.float64),
        "outer_tr": np.array([ ow, -oh, z], np.float64),
        "outer_br": np.array([ ow,  oh, z], np.float64),
        "outer_bl": np.array([-ow,  oh, z], np.float64),
        "inner_tl": np.array([-iw, -ih, z], np.float64),
        "inner_tr": np.array([ iw, -ih, z], np.float64),
        "inner_br": np.array([ iw,  ih, z], np.float64),
        "inner_bl": np.array([-iw,  ih, z], np.float64),
    }


def base_gate_R_world() -> np.ndarray:
    # gate local x=right, y=down, z=through gate; world x=forward,y=left,z=up
    return np.array([
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ], dtype=np.float64)


def make_gate_T_world(position_xyz, roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0) -> np.ndarray:
    R0 = base_gate_R_world()
    Rlocal = Rotation.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R0 @ Rlocal
    T[:3, 3] = np.asarray(position_xyz, dtype=np.float64)
    return T


def look_at_T_world_camera(position, target, world_up=np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    p = np.asarray(position, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    z = t - p
    z /= max(np.linalg.norm(z), 1e-12)
    x = np.cross(z, world_up)
    if np.linalg.norm(x) < 1e-7:
        world_up = np.array([0.0, 1.0, 0.0])
        x = np.cross(z, world_up)
    x /= max(np.linalg.norm(x), 1e-12)
    y = np.cross(z, x)
    y /= max(np.linalg.norm(y), 1e-12)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = p
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    return (T @ h.T).T[:, :3]


def project_points(K: np.ndarray, pts_cam: np.ndarray):
    pts = np.asarray(pts_cam, dtype=np.float64)
    z = pts[:, 2]
    uv = np.full((len(pts), 2), np.nan, dtype=np.float64)
    valid = z > 1e-8
    if np.any(valid):
        p = pts[valid]
        uv[valid, 0] = K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2]
        uv[valid, 1] = K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]
    return uv, valid


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3 or not np.isfinite(poly).all():
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def rotation_error_components_deg(T_camera_gate: np.ndarray) -> dict:
    R = T_camera_gate[:3, :3]
    e = Rotation.from_matrix(R).as_euler("xyz", degrees=True)
    return {"roll": float(e[0]), "pitch": float(e[1]), "yaw": float(e[2])}
