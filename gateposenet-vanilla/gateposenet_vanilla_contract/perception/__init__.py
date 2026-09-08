"""Perception front-end: GateNet -> QuAdGate corner detection -> PnP pose.

These modules consume GateNet's binary segmentation masks and reproduce the
geometric perception stage of MonoRace (Materials & Methods > Perception).
"""

from .quadgate import QuAdGate, GateCornerPrior, detect_lines
from .pnp import GateGeometry, solve_pnp_gates, CameraIntrinsics
from .pose import (GatePose, clean_mask, quad_from_mask, pose_from_corners,
                   gate_pose_from_mask, normal_angle_error_deg)

__all__ = [
    "QuAdGate",
    "GateCornerPrior",
    "detect_lines",
    "GateGeometry",
    "solve_pnp_gates",
    "CameraIntrinsics",
    "GatePose",
    "clean_mask",
    "quad_from_mask",
    "pose_from_corners",
    "gate_pose_from_mask",
    "normal_angle_error_deg",
]
