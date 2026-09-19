from __future__ import annotations
import math
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from .geometry import make_gate_T_world_course, look_at_T_world_camera


def _choice_from_prob_dict(rng: np.random.Generator, d: dict) -> str:
    names = list(d.keys())
    p = np.asarray([float(d[n]) for n in names], dtype=np.float64)
    p /= p.sum()
    return str(rng.choice(names, p=p))


def _course_style_cfg(cfg: dict, style: str) -> dict:
    styles = cfg.get("course_styles", {})
    if style in styles:
        return styles[style]
    return {"turn_deg": [-25.0, 25.0], "vertical_step_m": [-0.35, 0.35]}


def generate_course(rng: np.random.Generator, cfg: dict):
    """Generate a 3-D gate course with globally varying direction.

    Unlike the original generator, gates are not constrained to monotonically
    increasing world X. A sequence samples a course style and performs a smooth-ish
    random walk in heading and altitude. Each gate is oriented relative to the
    local course direction plus independent yaw/pitch/roll perturbations.
    """
    n = int(rng.integers(cfg["gates_min"], cfg["gates_max"] + 1))

    style_probs = cfg.get("course_style_probabilities", {
        "straight": 0.10,
        "flowing": 0.45,
        "technical": 0.30,
        "aggressive": 0.15,
    })
    style = _choice_from_prob_dict(rng, style_probs)
    style_cfg = _course_style_cfg(cfg, style)

    heading = float(rng.uniform(-math.pi, math.pi))
    z_min, z_max = cfg.get("gate_height_m", [1.0, 2.5])
    pos = np.array([0.0, 0.0, float(rng.uniform(z_min, z_max))], dtype=np.float64)
    gates = []

    turn_range = style_cfg.get("turn_deg", [-25.0, 25.0])
    vertical_step = style_cfg.get("vertical_step_m", [-0.35, 0.35])

    for i in range(n):
        if i > 0:
            turn_deg = float(rng.uniform(*turn_range))
            # Occasional stronger turn for technical/aggressive courses.
            if style in {"technical", "aggressive"} and rng.random() < float(style_cfg.get("hard_turn_probability", 0.20)):
                hard = style_cfg.get("hard_turn_deg", [-65.0, 65.0])
                turn_deg = float(rng.uniform(*hard))
            heading += math.radians(turn_deg)
        else:
            turn_deg = 0.0

        spacing = float(rng.uniform(*cfg["course_spacing_m"]))
        pos = pos.copy()
        pos[0] += math.cos(heading) * spacing
        pos[1] += math.sin(heading) * spacing
        pos[2] = float(np.clip(pos[2] + rng.uniform(*vertical_step), z_min, z_max))

        roll = float(rng.uniform(*cfg["gate_roll_deg"]))
        pitch = float(rng.uniform(*cfg["gate_pitch_deg"]))
        yaw_offset = float(rng.uniform(*cfg["gate_yaw_deg"]))
        T = make_gate_T_world_course(pos, heading, roll, pitch, yaw_offset)

        gates.append({
            "track_id": f"gate_{i + 1:04d}",
            "gate_type_id": "standard_gate",
            "route_order_index": i,
            "T_world_gate": T,
            "position_world": pos.copy(),
            "course_heading_rad": float(heading),
            "course_heading_deg": float(math.degrees(heading)),
            "turn_from_previous_deg": float(turn_deg),
            "gate_yaw_offset_deg": yaw_offset,
            "gate_pitch_deg": pitch,
            "gate_roll_deg": roll,
            "course_style": style,
        })

    return gates


def _build_spatial_path(rng: np.random.Generator, gates: list[dict], cfg: dict):
    gate_pts = np.asarray([g["position_world"] for g in gates], dtype=np.float64)
    headings = np.asarray([g["course_heading_rad"] for g in gates], dtype=np.float64)

    approach = float(cfg.get("approach_distance_m", 5.0))
    exit_dist = float(cfg.get("exit_distance_m", 3.0))

    first_dir = np.array([math.cos(headings[0]), math.sin(headings[0]), 0.0])
    last_dir = np.array([math.cos(headings[-1]), math.sin(headings[-1]), 0.0])

    nodes = [gate_pts[0] - approach * first_dir]
    for p, h in zip(gate_pts, headings):
        lateral = np.array([-math.sin(h), math.cos(h), 0.0])
        lateral_off = float(rng.uniform(-cfg["path_lateral_offset_m"], cfg["path_lateral_offset_m"]))
        vertical_off = float(rng.uniform(-cfg["path_vertical_offset_m"], cfg["path_vertical_offset_m"]))
        nodes.append(p + lateral * lateral_off + np.array([0.0, 0.0, vertical_off]))
    nodes.append(gate_pts[-1] + exit_dist * last_dir)
    nodes = np.asarray(nodes, dtype=np.float64)

    # Chord-length parameterization avoids requiring monotonic world X.
    chord = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    u_nodes = np.concatenate([[0.0], np.cumsum(np.maximum(chord, 1e-4))])
    u_nodes /= u_nodes[-1]

    splines = [CubicSpline(u_nodes, nodes[:, ax], bc_type="natural") for ax in range(3)]
    dense_n = max(int(cfg.get("path_dense_samples", 4096)), 512)
    u_dense = np.linspace(0.0, 1.0, dense_n)
    p_dense = np.stack([s(u_dense) for s in splines], axis=1)
    dp_dense = np.stack([s(u_dense, 1) for s in splines], axis=1)

    ds = np.linalg.norm(np.diff(p_dense, axis=0), axis=1)
    s_dense = np.concatenate([[0.0], np.cumsum(ds)])
    total_length = float(s_dense[-1])

    # Gate arc-length locations come from the corresponding spline waypoint.
    # This preserves route ordering even if a technical course crosses itself.
    gate_s = []
    for i, g in enumerate(gates):
        u_gate = float(u_nodes[i + 1])
        sg = float(np.interp(u_gate, u_dense, s_dense))
        g["path_s_m"] = sg
        gate_s.append(sg)

    return {
        "u_dense": u_dense,
        "p_dense": p_dense,
        "dp_dense": dp_dense,
        "s_dense": s_dense,
        "total_length_m": total_length,
        "gate_s_m": np.asarray(gate_s, dtype=np.float64),
    }


def _interp_path(path: dict, s_query: float):
    s_dense = path["s_dense"]
    u_dense = path["u_dense"]
    p_dense = path["p_dense"]
    dp_dense = path["dp_dense"]
    sq = float(np.clip(s_query, 0.0, s_dense[-1]))
    u = float(np.interp(sq, s_dense, u_dense))
    # Interpolate position and derivative component-wise over u.
    pos = np.array([np.interp(u, u_dense, p_dense[:, a]) for a in range(3)], dtype=np.float64)
    tangent = np.array([np.interp(u, u_dense, dp_dense[:, a]) for a in range(3)], dtype=np.float64)
    tangent /= max(np.linalg.norm(tangent), 1e-9)
    return pos, tangent


def _path_curvature(path: dict):
    p = path["p_dense"]
    s = path["s_dense"]
    d1 = np.gradient(p, axis=0)
    norm = np.linalg.norm(d1, axis=1, keepdims=True)
    t = d1 / np.maximum(norm, 1e-9)
    dt = np.gradient(t, axis=0)
    ds_du = np.gradient(s)
    kappa = np.linalg.norm(dt, axis=1) / np.maximum(ds_du, 1e-5)
    return np.clip(kappa, 0.0, 5.0)


def _sample_motion_profile(rng: np.random.Generator, cfg: dict):
    probs = cfg.get("profile_probabilities", {
        "slow": 0.15,
        "medium": 0.35,
        "fast": 0.35,
        "aggressive": 0.15,
    })
    name = _choice_from_prob_dict(rng, probs)
    profile = dict(cfg["profiles"][name])
    profile["name"] = name
    profile["cruise_speed_mps"] = float(rng.uniform(*profile["cruise_speed_mps_range"]))
    profile["accel_limit_mps2"] = float(rng.uniform(*profile.get("accel_limit_mps2_range", [2.0, 4.0])))
    profile["speed_variation_fraction"] = float(rng.uniform(*profile.get("speed_variation_fraction_range", [0.08, 0.22])))
    profile["orientation_jitter_deg"] = float(rng.uniform(*profile.get("orientation_jitter_deg_range", [3.0, 8.0])))
    return profile


def _target_speed_function(rng: np.random.Generator, path: dict, profile: dict, motion_cfg: dict):
    s_dense = path["s_dense"]
    total = max(float(s_dense[-1]), 1e-6)
    sn = s_dense / total
    base = float(profile["cruise_speed_mps"])
    amp = float(profile["speed_variation_fraction"])

    c1 = float(rng.uniform(0.6, 1.8))
    c2 = float(rng.uniform(1.8, 4.0))
    p1 = float(rng.uniform(0, 2 * math.pi))
    p2 = float(rng.uniform(0, 2 * math.pi))
    v = base * (1.0 + 0.70 * amp * np.sin(2 * math.pi * c1 * sn + p1) + 0.30 * amp * np.sin(2 * math.pi * c2 * sn + p2))

    # Explicit acceleration/deceleration events along the sequence.
    events = int(rng.integers(*motion_cfg.get("speed_event_count_range", [1, 4])))
    event_meta = []
    for _ in range(events):
        center = float(rng.uniform(0.12, 0.88))
        width = float(rng.uniform(0.04, 0.14))
        magnitude = float(rng.uniform(-amp, amp))
        # Ensure a useful mix of braking and acceleration events.
        if abs(magnitude) < 0.04:
            magnitude = 0.04 if rng.random() < 0.5 else -0.04
        bump = np.exp(-0.5 * ((sn - center) / width) ** 2)
        v *= (1.0 + magnitude * bump)
        event_meta.append({"center_fraction": center, "width_fraction": width, "fractional_speed_change": magnitude})

    # Slow down for high-curvature parts of the path.
    curvature = _path_curvature(path)
    turn_strength = float(profile.get("turn_slowdown_strength", 2.0))
    v /= (1.0 + turn_strength * curvature)

    vmin = float(profile.get("min_speed_mps", max(0.5, 0.45 * base)))
    vmax = float(profile.get("max_speed_mps", 1.45 * base))
    v = np.clip(v, vmin, vmax)
    return v, event_meta


def _angular_velocity_local(T_prev: np.ndarray, T_cur: np.ndarray, dt: float):
    R_prev = T_prev[:3, :3]
    R_cur = T_cur[:3, :3]
    R_rel = R_prev.T @ R_cur
    rotvec = Rotation.from_matrix(R_rel).as_rotvec()
    return rotvec / max(dt, 1e-9)


def generate_camera_trajectory(rng: np.random.Generator, gates: list[dict], trajectory_cfg: dict, motion_cfg: dict, fps: float):
    """Generate physically time-sampled flight states at fixed FPS.

    Position is sampled by integrating an explicit speed profile over arc length,
    so `speed_mps`, velocity, and acceleration are meaningful in real units.
    """
    path_cfg = trajectory_cfg
    path = _build_spatial_path(rng, gates, path_cfg)
    profile = _sample_motion_profile(rng, motion_cfg)
    target_dense, speed_events = _target_speed_function(rng, path, profile, motion_cfg)

    dt = 1.0 / float(fps)
    total = float(path["total_length_m"])
    s = 0.0
    start_frac = float(profile.get("start_speed_fraction", 0.75))
    v = max(float(profile.get("min_speed_mps", 0.5)), float(profile["cruise_speed_mps"]) * start_frac)
    accel_limit = float(profile["accel_limit_mps2"])

    samples = []
    max_frames = int(motion_cfg.get("max_frames_per_sequence", 900))
    fi = 0
    while s < total - 1e-5 and fi < max_frames:
        target_v = float(np.interp(s, path["s_dense"], target_dense))
        dv = np.clip(target_v - v, -accel_limit * dt, accel_limit * dt)
        v = max(0.25, v + float(dv))
        pos, tangent = _interp_path(path, s)
        samples.append({"path_s_m": float(s), "position_world_m": pos, "tangent_world": tangent, "speed_mps": float(v), "target_speed_mps": target_v})
        s = min(total, s + v * dt)
        fi += 1

    truncated_by_max_frames = bool(fi >= max_frames and s < total - 1e-5)
    # Only snap to the exact path end when the integrated trajectory actually
    # reached it. Never introduce a synthetic teleport when max_frames truncates.
    if not truncated_by_max_frames and (not samples or samples[-1]["path_s_m"] < total - 1e-4):
        pos, tangent = _interp_path(path, total)
        samples.append({"path_s_m": total, "position_world_m": pos, "tangent_world": tangent, "speed_mps": float(v), "target_speed_mps": float(np.interp(total, path["s_dense"], target_dense))})

    phase_y = float(rng.uniform(0, 2 * math.pi))
    phase_z = float(rng.uniform(0, 2 * math.pi))
    gaze_cycles = float(rng.uniform(0.6, 2.2))
    jitter_deg = float(profile["orientation_jitter_deg"])
    target_blend = float(path_cfg.get("target_gate_look_blend", 0.55))
    lookahead = float(path_cfg.get("camera_lookahead_m", 4.0))
    gate_s = path["gate_s_m"]

    # First pass: camera orientation and velocity.
    for i, st in enumerate(samples):
        s_now = st["path_s_m"]
        pos = st["position_world_m"]
        tangent = st["tangent_world"]
        tnorm = s_now / max(total, 1e-9)

        ahead_idx = np.where(gate_s >= s_now - float(path_cfg.get("gate_pass_margin_m", 0.4)))[0]
        current_idx = int(ahead_idx[0]) if len(ahead_idx) else None
        path_aim = pos + tangent * lookahead
        if current_idx is not None:
            gate_aim = gates[current_idx]["position_world"]
            aim = (1.0 - target_blend) * path_aim + target_blend * gate_aim
            st["current_target_route_index"] = current_idx
        else:
            aim = path_aim
            st["current_target_route_index"] = None

        horizontal = tangent.copy()
        horizontal[2] = 0.0
        horizontal /= max(np.linalg.norm(horizontal), 1e-9)
        lateral = np.array([-horizontal[1], horizontal[0], 0.0])
        yaw_offset = math.tan(math.radians(jitter_deg * 0.55 * math.sin(2 * math.pi * gaze_cycles * tnorm + phase_y))) * lookahead
        pitch_offset = math.tan(math.radians(jitter_deg * 0.35 * math.sin(2 * math.pi * 0.73 * gaze_cycles * tnorm + phase_z))) * lookahead
        aim = aim + lateral * yaw_offset + np.array([0.0, 0.0, pitch_offset])
        T_wc = look_at_T_world_camera(pos, aim)
        st["T_world_camera"] = T_wc
        st["velocity_world_mps"] = tangent * st["speed_mps"]

    # Second pass: physically meaningful acceleration and angular velocity.
    for i, st in enumerate(samples):
        if len(samples) == 1:
            acc = np.zeros(3, dtype=np.float64)
            omega = np.zeros(3, dtype=np.float64)
        else:
            if i == 0:
                acc = (samples[1]["velocity_world_mps"] - st["velocity_world_mps"]) / dt
                omega = _angular_velocity_local(st["T_world_camera"], samples[1]["T_world_camera"], dt)
            else:
                acc = (st["velocity_world_mps"] - samples[i - 1]["velocity_world_mps"]) / dt
                omega = _angular_velocity_local(samples[i - 1]["T_world_camera"], st["T_world_camera"], dt)
        st["acceleration_world_mps2"] = acc
        st["angular_velocity_camera_radps"] = omega
        st["frame_index"] = i
        st["time_s"] = i * dt

    meta = {
        "motion_profile": profile,
        "speed_events": speed_events,
        "path_length_m": total,
        "num_frames": len(samples),
        "duration_s": (len(samples) - 1) * dt if samples else 0.0,
        "speed_min_mps": float(min(s["speed_mps"] for s in samples)),
        "speed_max_mps": float(max(s["speed_mps"] for s in samples)),
        "speed_mean_mps": float(np.mean([s["speed_mps"] for s in samples])),
        "course_style": gates[0].get("course_style") if gates else None,
        "completed_course": not truncated_by_max_frames,
        "truncated_by_max_frames": truncated_by_max_frames,
        "traversed_length_m": float(samples[-1]["path_s_m"]) if samples else 0.0,
    }
    return samples, meta
