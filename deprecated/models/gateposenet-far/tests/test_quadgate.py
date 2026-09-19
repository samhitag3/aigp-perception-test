"""Smoke tests for QuAdGate corner detection and PnP geometry."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perception.pnp import (CameraIntrinsics, GateGeometry, gate_corners_world,
                            solve_pnp_gates)
from perception.quadgate import QuAdGate, detect_lines


def _square_mask(size=256, inset=60, thickness=10):
    """A hollow square gate mask (frame only)."""
    m = np.zeros((size, size), np.uint8)
    a, b = inset, size - inset
    cv2.rectangle(m, (a, a), (b, b), 255, thickness)
    return m


def test_detect_lines():
    m = _square_mask()
    lines = detect_lines(m)
    assert lines.shape[1] == 4
    assert len(lines) >= 4  # at least the four sides


def test_quadgate_finds_corners():
    m = _square_mask()
    quad = QuAdGate(threshold=0.5)
    cands = quad.candidates(m)
    assert len(cands) > 0
    # at least one candidate near each true outer corner (a,a)/(b,b)...
    a, b = 60, 256 - 60
    corners = np.array([[a, a], [b, a], [b, b], [a, b]], float)
    found = np.array([c["xy"] for c in cands])
    for corner in corners:
        d = np.linalg.norm(found - corner, axis=1).min()
        assert d < 25, f"no candidate near {corner} (min dist {d:.1f})"


def test_pnp_roundtrip():
    """Project known gate corners with a known pose, then recover it via PnP."""
    intr = CameraIntrinsics(fx=300, fy=300, cx=192, cy=192)
    geom = GateGeometry()
    gate_pos = np.array([5.0, 0.0, 0.0])
    gate_yaw = 0.0
    world = gate_corners_world(geom, gate_pos, gate_yaw)  # (8,3)

    # camera at origin looking +x: build rvec/tvec and project
    # world->cam rotation: x_world maps to z_cam (optical axis)
    R_cw = np.array([[0, 1, 0],
                     [0, 0, 1],
                     [1, 0, 0]], float)
    rvec, _ = cv2.Rodrigues(R_cw)
    cam_pos = np.array([0.0, 0.0, 0.0])
    tvec = (-R_cw @ cam_pos).reshape(3, 1)
    img_pts, _ = cv2.projectPoints(world, rvec, tvec, intr.K, intr.D)
    img_pts = img_pts.reshape(-1, 2)

    obs = [{"image_points": img_pts, "corner_ids": list(range(8)),
            "gate_pos": gate_pos, "gate_yaw": gate_yaw}]
    res = solve_pnp_gates(obs, intr, geom, use_ransac=False)
    assert res["success"]
    assert res["reproj_err"] < 1.0
    assert np.linalg.norm(res["cam_pos_world"] - cam_pos) < 0.05


if __name__ == "__main__":
    test_detect_lines()
    test_quadgate_finds_corners()
    test_pnp_roundtrip()
    print("all quadgate/pnp tests passed")
