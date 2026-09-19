"""Camera-optical -> drone BODY-NED conversion (the DOWNSTREAM step).

The perception model (GatePoseNet) and all of its training data are purely
CAMERA-relative: gate position / orientation in the OpenCV optical frame
(+X right, +Y down, +Z forward). Where the drone's body is pointing is a
deployment concern — the camera is bolted to the frame at ONE known mount
angle, so the body-frame gate pose is a single fixed rotation away. This
module is that rotation, applied AFTER inference:

    camera image --(GatePoseNet)--> pose in CAMERA frame
                 --(this module + mount tilt)--> pose in BODY-NED

Frames
------
* camera optical (OpenCV): +X right, +Y down, +Z forward
* BODY-NED (MAVLink MAV_FRAME_BODY_NED): +X forward, +Y right, +Z down

The camera is pitched UP by the mount tilt, so the optical axis maps ABOVE
the body horizon (negative NED z). Hardware TPU mounts come in
30/35/40/45/50/60-degree detents (interpolated angles valid with a custom
mount); pass the angle of the mount actually on the frame.
"""

from __future__ import annotations

import numpy as np

# Printed TPU camera-mount detents (CameraMount - TPU (NN).stl, Nov 2025).
MOUNT_TILT_DETENTS_DEG = (30.0, 35.0, 40.0, 45.0, 50.0, 60.0)
DEFAULT_MOUNT_TILT_DEG = 30.0

# Axis permutation: columns are the optical axes (X,Y,Z) written in BODY-NED.
#   optical X (right)   -> body Y (right)
#   optical Y (down)    -> body Z (down)
#   optical Z (forward) -> body X (forward)
_P = np.array(
    [[0.0, 0.0, 1.0],
     [1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


def optical_to_body(tilt_deg: float = DEFAULT_MOUNT_TILT_DEG) -> np.ndarray:
    """Rotation taking CAMERA-OPTICAL vectors into BODY-NED.

    ``R = Ry(+tilt) @ P``: the +tilt pitch about the body right axis lifts
    the optical forward ABOVE the body horizon (negative NED z), matching a
    camera physically pitched UP by ``tilt_deg``. Verified property::

        optical_to_body(t) @ [0,0,1] == [cos(t), 0, -sin(t)]
    """
    a = np.radians(float(tilt_deg))
    c, s = np.cos(a), np.sin(a)
    Ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
                  dtype=np.float64)
    return Ry @ _P


def body_to_optical(tilt_deg: float = DEFAULT_MOUNT_TILT_DEG) -> np.ndarray:
    """Inverse of :func:`optical_to_body`."""
    return optical_to_body(tilt_deg).T


def gate_pose_to_body(
    position_cam: np.ndarray,
    R_cam_gate: np.ndarray | None = None,
    tilt_deg: float = DEFAULT_MOUNT_TILT_DEG,
) -> dict:
    """Convert a camera-frame gate pose to BODY-NED.

    Parameters
    ----------
    position_cam : (3,) gate center in camera-optical meters (model output).
    R_cam_gate : optional (3, 3) gate rotation in the camera frame.
    tilt_deg : the mount angle physically on the frame.

    Returns dict with ``position_body`` (3,), ``distance_m``, and (when
    ``R_cam_gate`` given) ``R_body_gate`` and ``flythrough_axis_body`` —
    the direction to fly, in BODY-NED.
    """
    R_ob = optical_to_body(tilt_deg)
    p_cam = np.asarray(position_cam, dtype=np.float64).reshape(3)
    out: dict = {
        "position_body": R_ob @ p_cam,
        "distance_m": float(np.linalg.norm(p_cam)),
        "tilt_deg": float(tilt_deg),
    }
    if R_cam_gate is not None:
        R_bg = R_ob @ np.asarray(R_cam_gate, dtype=np.float64).reshape(3, 3)
        out["R_body_gate"] = R_bg
        out["flythrough_axis_body"] = R_bg[:, 2]
    return out
