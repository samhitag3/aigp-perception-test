"""Perspective-n-Point pose estimation from gate corners.

Reproduces the PnP stage of MonoRace (Materials & Methods > Perspective n
Points): corners detected on one *or more* gates are combined into a single PnP
optimisation. Using non-coplanar corners from gates at different depths improves
the translation/rotation disambiguation, and merging corners lets PnP reach its
>=4-point requirement more often.

Gate geometry (A2RL square gate): inner opening 1.5 m, outer 2.7 m. Corners are
expressed in each gate's frame and transformed to the world frame using the
known gate pose from the flight plan (map).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray = None  # distortion coeffs; None => images are undistorted

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]], np.float64)

    @property
    def D(self) -> np.ndarray:
        return np.zeros((4, 1)) if self.dist is None else np.asarray(self.dist, np.float64)

    @classmethod
    def nominal_square(cls, width: int, height: int):
        """SkyDreamer nominal pinhole intrinsics: fx=fy=(25/64)*W, c=center."""
        f = (25.0 / 64.0) * width
        return cls(fx=f, fy=(25.0 / 64.0) * height, cx=0.5 * width, cy=0.5 * height)


@dataclass
class GateGeometry:
    """Square gate corner model (metres), centred on the gate frame origin.

    Corner order: 0..3 inner (TL,TR,BR,BL), 4..7 outer (TL,TR,BR,BL).
    Gate plane is the local x=0 plane (y right, z down), matching a forward-
    looking racing gate; adjust ``plane`` if your convention differs.
    """
    inner: float = 1.5
    outer: float = 2.7

    def corners_gate_frame(self) -> np.ndarray:
        hi, ho = self.inner / 2.0, self.outer / 2.0
        # (y, z) in gate plane; x = 0
        pts2d = [(-hi, -hi), (hi, -hi), (hi, hi), (-hi, hi),
                 (-ho, -ho), (ho, -ho), (ho, ho), (-ho, ho)]
        return np.array([[0.0, y, z] for (y, z) in pts2d], np.float64)


def _rot_from_yaw(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)


def gate_corners_world(geom: GateGeometry, gate_pos, gate_yaw: float) -> np.ndarray:
    """Transform the 8 gate corners into the world frame."""
    R = _rot_from_yaw(gate_yaw)
    local = geom.corners_gate_frame()
    return (local @ R.T) + np.asarray(gate_pos, np.float64)[None, :]


def solve_pnp_gates(observations, intr: CameraIntrinsics, geom: GateGeometry = None,
                    min_points: int = 4, use_ransac: bool = True):
    """Solve a single PnP from corners across one or more gates.

    Args:
        observations: list of dicts, each:
            {
              "image_points": (N,2) float,    # detected pixel corners
              "corner_ids":   (N,) int in 0..7,
              "gate_pos":     (3,) world position of the gate,
              "gate_yaw":     float world yaw of the gate,
            }
        intr: camera intrinsics.
        geom: gate geometry (defaults to A2RL square gate).
    Returns dict with keys: success, rvec, tvec, R_wc, cam_pos_world, n_points,
    reproj_err — or {"success": False} if insufficient points.
    """
    geom = geom or GateGeometry()
    obj_pts, img_pts = [], []
    for obs in observations:
        world = gate_corners_world(geom, obs["gate_pos"], obs["gate_yaw"])
        for p, cid in zip(obs["image_points"], obs["corner_ids"]):
            obj_pts.append(world[int(cid)])
            img_pts.append(p)

    if len(obj_pts) < min_points:
        return {"success": False, "n_points": len(obj_pts)}

    obj_pts = np.asarray(obj_pts, np.float64)
    img_pts = np.asarray(img_pts, np.float64)

    flags = cv2.SOLVEPNP_ITERATIVE
    if use_ransac and len(obj_pts) >= 6:
        ok, rvec, tvec, _ = cv2.solvePnPRansac(
            obj_pts, img_pts, intr.K, intr.D, flags=flags, reprojectionError=5.0)
    else:
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, intr.K, intr.D, flags=flags)

    if not ok:
        return {"success": False, "n_points": len(obj_pts)}

    R_cw, _ = cv2.Rodrigues(rvec)              # world -> camera
    R_wc = R_cw.T                               # camera -> world
    cam_pos_world = (-R_wc @ tvec).ravel()      # camera centre in world

    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, intr.K, intr.D)
    reproj_err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1)))

    return {
        "success": True,
        "rvec": rvec, "tvec": tvec,
        "R_wc": R_wc,
        "cam_pos_world": cam_pos_world,
        "n_points": len(obj_pts),
        "reproj_err": reproj_err,
    }
