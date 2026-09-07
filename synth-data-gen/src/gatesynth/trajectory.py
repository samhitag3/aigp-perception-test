from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicSpline
from .geometry import make_gate_T_world, look_at_T_world_camera


def generate_course(rng: np.random.Generator, cfg: dict):
    n = int(rng.integers(cfg["gates_min"], cfg["gates_max"] + 1))
    spacing = rng.uniform(*cfg["course_spacing_m"], size=n)
    xs = np.cumsum(spacing)
    ys = rng.uniform(-cfg["lateral_gate_jitter_m"], cfg["lateral_gate_jitter_m"], size=n)
    zs = rng.uniform(1.1, 2.2, size=n)
    gates = []
    for i in range(n):
        roll = rng.uniform(*cfg["gate_roll_deg"])
        pitch = rng.uniform(*cfg["gate_pitch_deg"])
        yaw = rng.uniform(*cfg["gate_yaw_deg"])
        T = make_gate_T_world([xs[i], ys[i], zs[i]], roll, pitch, yaw)
        gates.append({
            "track_id": f"gate_{i+1:04d}",
            "gate_type_id": "standard_gate",
            "route_order_index": i,
            "T_world_gate": T,
            "position_world": np.array([xs[i], ys[i], zs[i]], np.float64),
        })
    return gates


def generate_camera_trajectory(rng: np.random.Generator, gates: list[dict], frames: int, cfg: dict):
    gate_pts = np.array([g["position_world"] for g in gates])
    x_nodes = np.concatenate([[gate_pts[0,0] - 4.0], gate_pts[:,0], [gate_pts[-1,0] + 2.0]])
    y_nodes = np.concatenate([[gate_pts[0,1]], gate_pts[:,1], [gate_pts[-1,1]]])
    z_nodes = np.concatenate([[gate_pts[0,2]], gate_pts[:,2], [gate_pts[-1,2]]])
    y_nodes += rng.uniform(-cfg["path_lateral_offset_m"], cfg["path_lateral_offset_m"], size=len(y_nodes))
    z_nodes += rng.uniform(-cfg["path_vertical_offset_m"], cfg["path_vertical_offset_m"], size=len(z_nodes))
    # ensure strictly increasing x nodes
    t_nodes = (x_nodes - x_nodes[0]) / (x_nodes[-1] - x_nodes[0])
    csx = CubicSpline(t_nodes, x_nodes, bc_type="natural")
    csy = CubicSpline(t_nodes, y_nodes, bc_type="natural")
    csz = CubicSpline(t_nodes, z_nodes, bc_type="natural")
    ts = np.linspace(0.0, 1.0, frames)
    pos = np.stack([csx(ts), csy(ts), csz(ts)], axis=1)
    vel = np.stack([csx(ts, 1), csy(ts, 1), csz(ts, 1)], axis=1)

    phase_y = rng.uniform(0, 2*np.pi)
    phase_z = rng.uniform(0, 2*np.pi)
    jitter_deg = cfg["orientation_jitter_deg"]
    out = []
    for i, t in enumerate(ts):
        d = vel[i]
        d = d / max(np.linalg.norm(d), 1e-9)
        # smooth gaze perturbation to create changing perspective
        dy = np.tan(np.deg2rad(jitter_deg * 0.35 * np.sin(2*np.pi*t + phase_y)))
        dz = np.tan(np.deg2rad(jitter_deg * 0.25 * np.sin(1.5*np.pi*t + phase_z)))
        aim = pos[i] + d * 4.0 + np.array([0.0, dy, dz])
        T_wc = look_at_T_world_camera(pos[i], aim)
        out.append((pos[i], vel[i], T_wc))
    return out
