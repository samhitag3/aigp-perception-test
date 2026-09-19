from __future__ import annotations
import math
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation


FT_TO_M = 0.3048
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _choice_from_prob_dict(rng: np.random.Generator, d: dict) -> str:
    names = list(d.keys())
    p = np.asarray([float(d[n]) for n in names], dtype=np.float64)
    p /= p.sum()
    return str(rng.choice(names, p=p))


def _rotation_deg_to_flight_dir(rotation_deg: float) -> np.ndarray:
    """
    PDF convention:
      0°   = toward top edge (decreasing map Y)
      90°  = right (+X on map)
      180° = down (+Y on map)
      270° = left (-X on map)

    World convention used here:
      world_x = +map_x * 0.3048
      world_y = -map_y * 0.3048
      world_z = up

    Therefore:
      0°   -> +world_y
      90°  -> +world_x
      180° -> -world_y
      270° -> -world_x
    """
    th = math.radians(rotation_deg)
    return np.array([math.sin(th), math.cos(th), 0.0], dtype=np.float64)


def _right_axis_from_flight_dir(flight_dir: np.ndarray) -> np.ndarray:
    """Gate-local +X/right axis in world coordinates."""
    d = np.asarray(flight_dir, dtype=np.float64).copy()
    d[2] = 0.0
    d /= max(np.linalg.norm(d), 1e-9)
    right = np.cross(d, WORLD_UP)
    right /= max(np.linalg.norm(right), 1e-9)
    return right


def _gate_T_world_from_flight_dir(position_xyz, flight_dir_xy) -> np.ndarray:
    """
    Gate-local frame:
      +X = gate right
      +Y = gate down
      +Z = flight direction through the gate
    """
    pos = np.asarray(position_xyz, dtype=np.float64)
    z_axis = np.asarray(flight_dir_xy, dtype=np.float64).copy()
    z_axis[2] = 0.0
    z_axis /= max(np.linalg.norm(z_axis), 1e-9)

    x_axis = np.cross(z_axis, WORLD_UP)
    x_axis /= max(np.linalg.norm(x_axis), 1e-9)

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= max(np.linalg.norm(y_axis), 1e-9)

    R = np.column_stack([x_axis, y_axis, z_axis])
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def look_at_T_world_camera(position_world_m, target_world_m, up_world=(0.0, 0.0, 1.0)) -> np.ndarray:
    """
    Camera-optical convention:
      +X = right
      +Y = down
      +Z = forward
    """
    p = np.asarray(position_world_m, dtype=np.float64)
    t = np.asarray(target_world_m, dtype=np.float64)
    up = np.asarray(up_world, dtype=np.float64)

    z_cam = t - p
    if np.linalg.norm(z_cam) < 1e-9:
        z_cam = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    z_cam /= max(np.linalg.norm(z_cam), 1e-9)

    x_cam = np.cross(z_cam, up)
    if np.linalg.norm(x_cam) < 1e-7:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_cam = np.cross(z_cam, up)
    x_cam /= max(np.linalg.norm(x_cam), 1e-9)

    y_cam = np.cross(z_cam, x_cam)
    y_cam /= max(np.linalg.norm(y_cam), 1e-9)

    R = np.column_stack([x_cam, y_cam, z_cam])
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def generate_course(rng: np.random.Generator, cfg: dict):
    """Build the fixed real course, including the vertical Gate-9 double gate.

    Route order:
      gate_01 ... gate_08 -> double_gate_01_top ->
      double_gate_01_bottom -> gate_10

    The bottom opening uses the original Gate 9 center. The top opening is
    directly above it. Top is traversed in the PDF Gate-9 direction (180°),
    then the drone U-turns and traverses the bottom in the opposite direction.
    """
    if not cfg.get("use_fixed_track_map", False):
        raise ValueError("Fixed-course generator requires course.use_fixed_track_map: true")

    gate_center_z_m = float(cfg.get("gate_center_z_m", 1.35))
    position_jitter_m = float(cfg.get("position_jitter_m", 0.0))
    height_jitter_m = float(cfg.get("height_jitter_m", 0.0))
    yaw_jitter_deg = float(cfg.get("yaw_jitter_deg", 0.0))

    double_cfg = cfg.get("double_gate", {})
    double_enabled = bool(double_cfg.get("enabled", False))
    insert_after_gate = int(double_cfg.get("route_after_gate", len(cfg["track_gates"])))

    physical_gates = []

    for list_index, spec in enumerate(cfg["track_gates"], start=1):
        gate_name = spec.get("name", f"gate_{list_index:02d}")
        if not gate_name.startswith("gate_"):
            raise ValueError(f"Cannot infer route order from gate name: {gate_name}")

        actual_gate_number = int(gate_name.split("_")[-1])
        # Two route slots replace original Gate 9, so gate_10 is route index 10.
        if double_enabled and actual_gate_number > insert_after_gate:
            route_order_index = actual_gate_number
        else:
            route_order_index = actual_gate_number - 1

        x_m = float(spec["x_ft"]) * FT_TO_M
        y_m = -float(spec["y_ft"]) * FT_TO_M
        z_m = gate_center_z_m

        if position_jitter_m > 0:
            x_m += float(rng.uniform(-position_jitter_m, position_jitter_m))
            y_m += float(rng.uniform(-position_jitter_m, position_jitter_m))
        if height_jitter_m > 0:
            z_m += float(rng.uniform(-height_jitter_m, height_jitter_m))

        rotation_deg = float(spec["rotation_deg"])
        if yaw_jitter_deg > 0:
            rotation_deg += float(rng.uniform(-yaw_jitter_deg, yaw_jitter_deg))

        flight_dir = _rotation_deg_to_flight_dir(rotation_deg)
        T_world_gate = _gate_T_world_from_flight_dir([x_m, y_m, z_m], flight_dir)

        physical_gates.append({
            "track_id": gate_name,
            "gate_type_id": "standard_gate",
            "structure_id": gate_name,
            "structure_type": "single_gate",
            "structure_child": None,
            "is_route_target": True,
            "route_order_index": route_order_index,
            "T_world_gate": T_world_gate,
            "position_world": np.array([x_m, y_m, z_m], dtype=np.float64),
            "course_heading_deg": rotation_deg,
            "course_heading_rad": math.atan2(flight_dir[1], flight_dir[0]),
            "turn_from_previous_deg": 0.0,
            "gate_yaw_offset_deg": 0.0,
            "gate_pitch_deg": 0.0,
            "gate_roll_deg": 0.0,
            "course_style": "fixed_track_map",
            "map_x_ft": float(spec["x_ft"]),
            "map_y_ft": float(spec["y_ft"]),
            "map_rotation_deg": float(spec["rotation_deg"]),
        })

    if double_enabled:
        x_ft = float(double_cfg["x_ft"])
        y_ft = float(double_cfg["y_ft"])
        spacing_m = float(double_cfg["vertical_center_spacing_m"])
        top_rotation_deg = float(double_cfg.get("top_rotation_deg", 180.0))
        bottom_rotation_deg = float(double_cfg.get("bottom_rotation_deg", 0.0))

        bottom_center = np.array(
            [x_ft * FT_TO_M, -y_ft * FT_TO_M, gate_center_z_m], dtype=np.float64
        )
        top_center = bottom_center + np.array([0.0, 0.0, spacing_m], dtype=np.float64)

        for child_name, center, rotation_deg, route_idx in (
            ("top", top_center, top_rotation_deg, insert_after_gate),
            ("bottom", bottom_center, bottom_rotation_deg, insert_after_gate + 1),
        ):
            flight_dir = _rotation_deg_to_flight_dir(rotation_deg)
            physical_gates.append({
                "track_id": f"double_gate_01_{child_name}",
                "gate_type_id": "standard_gate",
                "structure_id": str(double_cfg.get("structure_id", "double_gate_01")),
                "structure_type": "double_gate",
                "structure_child": child_name,
                "is_route_target": True,
                "route_order_index": route_idx,
                "T_world_gate": _gate_T_world_from_flight_dir(center, flight_dir),
                "position_world": center,
                "course_heading_deg": rotation_deg,
                "course_heading_rad": math.atan2(flight_dir[1], flight_dir[0]),
                "turn_from_previous_deg": 0.0,
                "gate_yaw_offset_deg": 0.0,
                "gate_pitch_deg": 0.0,
                "gate_roll_deg": 0.0,
                "course_style": "fixed_track_map",
                "map_x_ft": x_ft,
                "map_y_ft": y_ft,
                "map_rotation_deg": rotation_deg,
                "vertical_center_spacing_m": spacing_m,
            })

    physical_gates.sort(key=lambda g: g["route_order_index"])

    previous_heading = None
    for g in physical_gates:
        current = float(g["course_heading_deg"])
        if previous_heading is None:
            turn = 0.0
        else:
            turn = current - previous_heading
            while turn > 180.0:
                turn -= 360.0
            while turn < -180.0:
                turn += 360.0
        g["turn_from_previous_deg"] = float(turn)
        previous_heading = current

    return physical_gates


def _sample_success_crossing_point(
    rng: np.random.Generator,
    gate: dict,
    cfg: dict,
    shared_lateral: float | None = None,
) -> np.ndarray:
    """Pick a point safely inside the 1.5 m opening."""
    center = np.asarray(gate["position_world"], dtype=np.float64).copy()
    d = _rotation_deg_to_flight_dir(float(gate["course_heading_deg"]))
    right = _right_axis_from_flight_dir(d)

    half_opening = 0.75
    safety_margin = float(cfg.get("gate_crossing_safety_margin_m", 0.22))
    max_safe = max(0.05, half_opening - safety_margin)
    lat_lim = min(float(cfg.get("gate_crossing_lateral_jitter_m", 0.22)), max_safe)
    vert_lim = min(float(cfg.get("gate_crossing_vertical_jitter_m", 0.16)), max_safe)

    lateral = shared_lateral if shared_lateral is not None else float(rng.uniform(-lat_lim, lat_lim))
    vertical = float(rng.uniform(-vert_lim, vert_lim))
    return center + right * lateral + WORLD_UP * vertical


def _sample_collision_point(rng: np.random.Generator, gate: dict, failure_cfg: dict) -> np.ndarray:
    """Choose a point on the visible gate frame rather than inside its opening."""
    center = np.asarray(gate["position_world"], dtype=np.float64).copy()
    d = _rotation_deg_to_flight_dir(float(gate["course_heading_deg"]))
    right = _right_axis_from_flight_dir(d)

    impact = failure_cfg.get("impact", {})
    ring_radius = float(impact.get("ring_center_offset_m", 1.05))
    cross_jitter = float(impact.get("along_bar_jitter_m", 0.22))
    side = str(rng.choice(["left", "right", "top", "bottom"]))

    if side == "left":
        point = center - ring_radius * right + WORLD_UP * float(rng.uniform(-cross_jitter, cross_jitter))
    elif side == "right":
        point = center + ring_radius * right + WORLD_UP * float(rng.uniform(-cross_jitter, cross_jitter))
    elif side == "top":
        point = center + ring_radius * WORLD_UP + right * float(rng.uniform(-cross_jitter, cross_jitter))
    else:
        point = center - ring_radius * WORLD_UP + right * float(rng.uniform(-cross_jitter, cross_jitter))

    return point


def _append_gate_success_nodes(
    nodes: list[np.ndarray],
    gate_node_indices: list[int],
    gate: dict,
    crossing_point: np.ndarray,
    pre_standoff: float,
    post_standoff: float,
):
    d = _rotation_deg_to_flight_dir(float(gate["course_heading_deg"]))
    nodes.append(crossing_point - pre_standoff * d)
    gate_node_indices.append(len(nodes))
    nodes.append(crossing_point)
    nodes.append(crossing_point + post_standoff * d)


def _append_gate6_hairpin_success_nodes(
    nodes: list[np.ndarray],
    gate_node_indices: list[int],
    gate: dict,
    crossing_point: np.ndarray,
    cfg: dict,
):
    """Force the real Gate-6 hairpin shown on the supplied course map.

    Gate 5 is on the *front/+flight-direction* side of Gate 6.  Therefore simply
    adding a long pre-gate point is not sufficient: a spline from Gate 5 to that
    point can cross Gate 6's plane before reaching the approach side.

    The real racing line first goes around the map-top end of Gate 6, reaches the
    behind/approach side, turns back toward the gate, then crosses the opening in
    the +flight direction.
    """
    d = _rotation_deg_to_flight_dir(float(gate["course_heading_deg"]))
    right = _right_axis_from_flight_dir(d)

    # For Gate 6 (rotation 90 deg), -right is toward the top of the PDF/map.
    map_top = -right

    front_standoff = float(cfg.get("gate6_front_bypass_standoff_m", 2.0))
    behind_standoff = float(cfg.get("gate6_behind_bypass_standoff_m", 5.0))
    lateral_clearance = float(cfg.get("gate6_bypass_lateral_clearance_m", 2.4))
    approach_standoff = float(cfg.get("gate6_approach_standoff_m", 2.4))
    exit_standoff = float(cfg.get("gate6_exit_standoff_m", 2.0))

    # 1) Stay on the front side, but move above the physical gate span.
    front_bypass = (
        crossing_point
        + front_standoff * d
        + lateral_clearance * map_top
    )

    # 2) Cross the gate plane OUTSIDE the 2.7 m-wide physical frame.
    plane_bypass = (
        crossing_point
        + lateral_clearance * map_top
    )

    # 3) Continue well behind Gate 6, still outside the frame.
    behind_bypass = (
        crossing_point
        - behind_standoff * d
        + lateral_clearance * map_top
    )

    # 4) Curl back toward the centerline on the correct approach side.
    pre_gate = crossing_point - approach_standoff * d

    nodes.extend([
        front_bypass,
        plane_bypass,
        behind_bypass,
        pre_gate,
    ])

    # 5) Now pass through the actual opening and continue toward Gate 7.
    gate_node_indices.append(len(nodes))
    nodes.append(crossing_point)
    nodes.append(crossing_point + exit_standoff * d)


def _append_gate6_hairpin_failure_nodes(
    rng: np.random.Generator,
    nodes: list[np.ndarray],
    gate_node_indices: list[int],
    gate: dict,
    cfg: dict,
    failure_cfg: dict,
):
    """Use the same real Gate-6 approach, then intentionally hit its frame."""
    d = _rotation_deg_to_flight_dir(float(gate["course_heading_deg"]))
    right = _right_axis_from_flight_dir(d)
    map_top = -right

    front_standoff = float(cfg.get("gate6_front_bypass_standoff_m", 2.0))
    behind_standoff = float(cfg.get("gate6_behind_bypass_standoff_m", 5.0))
    lateral_clearance = float(cfg.get("gate6_bypass_lateral_clearance_m", 2.4))

    impact_cfg = failure_cfg.get("impact", {})
    impact_standoff = float(
        impact_cfg.get(
            "pre_impact_standoff_m",
            cfg.get("gate6_approach_standoff_m", 2.4),
        )
    )

    center = np.asarray(gate["position_world"], dtype=np.float64)
    collision = _sample_collision_point(rng, gate, failure_cfg)

    # Bypass uses the gate-center altitude/geometry so the path definitely goes
    # around the map-top end of the physical frame before getting behind it.
    front_bypass = center + front_standoff * d + lateral_clearance * map_top
    plane_bypass = center + lateral_clearance * map_top
    behind_bypass = center - behind_standoff * d + lateral_clearance * map_top
    pre_impact = collision - impact_standoff * d

    nodes.extend([front_bypass, plane_bypass, behind_bypass, pre_impact])

    gate_node_indices.append(len(nodes))
    nodes.append(collision)

    penetration = float(impact_cfg.get("penetration_m", 0.08))
    drop_distance = float(impact_cfg.get("drop_distance_m", 1.25))
    drop_forward = float(impact_cfg.get("drop_forward_m", 0.25))
    min_final_z = float(impact_cfg.get("min_final_z_m", 0.12))
    drop_to_ground = bool(impact_cfg.get("drop_to_ground", True))

    nodes.append(collision + penetration * d - 0.18 * WORLD_UP)
    final = collision + drop_forward * d
    final[2] = (
        min_final_z
        if drop_to_ground
        else max(min_final_z, collision[2] - drop_distance)
    )
    nodes.append(final)


def _append_failure_nodes(
    rng: np.random.Generator,
    nodes: list[np.ndarray],
    gate_node_indices: list[int],
    gate: dict,
    cfg: dict,
    failure_cfg: dict,
):
    impact_cfg = failure_cfg.get("impact", {})
    d = _rotation_deg_to_flight_dir(float(gate["course_heading_deg"]))
    collision = _sample_collision_point(rng, gate, failure_cfg)
    standoff = float(impact_cfg.get("pre_impact_standoff_m", cfg.get("gate_crossing_standoff_m", 1.8)))
    penetration = float(impact_cfg.get("penetration_m", 0.08))
    drop_distance = float(impact_cfg.get("drop_distance_m", 1.25))
    drop_forward = float(impact_cfg.get("drop_forward_m", 0.25))
    min_final_z = float(impact_cfg.get("min_final_z_m", 0.12))
    drop_to_ground = bool(impact_cfg.get("drop_to_ground", True))

    nodes.append(collision - standoff * d)
    gate_node_indices.append(len(nodes))
    nodes.append(collision)

    # A very short continuation through the contact plane, then a drop.
    after_hit = collision + penetration * d - 0.18 * WORLD_UP
    nodes.append(after_hit)
    final = collision + drop_forward * d
    final[2] = min_final_z if drop_to_ground else max(min_final_z, collision[2] - drop_distance)
    nodes.append(final)


def _build_spatial_path(
    rng: np.random.Generator,
    gates: list[dict],
    cfg: dict,
    failure_plan: dict | None = None,
    failure_cfg: dict | None = None,
):
    """Build a path that deliberately crosses every successful gate.

    Every normal gate gets pre-center-post control points aligned with its flight
    direction. Gate 6 gets a longer setup point behind the gate, which forces
    Gate 5 -> behind Gate 6 -> through Gate 6 -> toward Gate 7.

    The vertical double gate receives a dedicated top-pass/U-turn/bottom-pass
    maneuver. If the sequence is an intentional failure, the selected gate is
    replaced by a frame collision followed by a short drop and the path ends.
    """
    failure_cfg = failure_cfg or {}
    route_gates = sorted(
        [g for g in gates if g.get("is_route_target", True)],
        key=lambda g: g["route_order_index"],
    )
    if not route_gates:
        raise ValueError("No route gates were generated")

    for g in route_gates:
        g["path_s_m"] = None

    failure_track_id = failure_plan.get("target_track_id") if failure_plan else None
    if failure_track_id is not None and failure_track_id not in {g["track_id"] for g in route_gates}:
        raise ValueError(f"Unknown failure target track_id: {failure_track_id}")

    approach_distance = float(cfg.get("approach_distance_m", 4.5))
    exit_distance = float(cfg.get("exit_distance_m", 3.0))
    standoff = float(cfg.get("gate_crossing_standoff_m", 1.8))

    double_standoff = float(cfg.get("double_gate_crossing_standoff_m", 2.0))
    turnaround_extra = float(cfg.get("double_gate_turnaround_extra_m", 2.5))
    turnaround_lateral = float(cfg.get("double_gate_turnaround_lateral_m", 2.5))

    # Success crossing points stay comfortably inside each opening.
    crossing_points: dict[str, np.ndarray] = {}
    shared_double_lateral = float(
        rng.uniform(
            -min(float(cfg.get("gate_crossing_lateral_jitter_m", 0.22)), 0.18),
            min(float(cfg.get("gate_crossing_lateral_jitter_m", 0.22)), 0.18),
        )
    )
    for g in route_gates:
        shared = shared_double_lateral if g.get("structure_type") == "double_gate" else None
        crossing_points[g["track_id"]] = _sample_success_crossing_point(rng, g, cfg, shared_lateral=shared)

    first_gate = route_gates[0]
    first_dir = _rotation_deg_to_flight_dir(float(first_gate["course_heading_deg"]))
    nodes: list[np.ndarray] = [crossing_points[first_gate["track_id"]] - approach_distance * first_dir]
    gate_node_indices: list[int] = []
    active_route_gates: list[dict] = []

    turn_side = 1.0 if rng.random() < 0.5 else -1.0
    i = 0
    stopped_for_failure = False

    while i < len(route_gates):
        g = route_gates[i]
        track_id = g["track_id"]

        is_double_top = (
            g.get("structure_type") == "double_gate"
            and g.get("structure_child") == "top"
        )
        has_matching_bottom = (
            i + 1 < len(route_gates)
            and route_gates[i + 1].get("structure_type") == "double_gate"
            and route_gates[i + 1].get("structure_child") == "bottom"
            and route_gates[i + 1].get("structure_id") == g.get("structure_id")
        )

        if is_double_top and has_matching_bottom:
            bottom = route_gates[i + 1]
            top_point = crossing_points[g["track_id"]]
            bottom_point = crossing_points[bottom["track_id"]]
            top_dir = _rotation_deg_to_flight_dir(float(g["course_heading_deg"]))
            bottom_dir = _rotation_deg_to_flight_dir(float(bottom["course_heading_deg"]))
            top_right = _right_axis_from_flight_dir(top_dir)

            if failure_track_id == g["track_id"]:
                active_route_gates.append(g)
                _append_failure_nodes(rng, nodes, gate_node_indices, g, cfg, failure_cfg)
                stopped_for_failure = True
                break

            # Successful top pass.
            nodes.append(top_point - double_standoff * top_dir)
            active_route_gates.append(g)
            gate_node_indices.append(len(nodes))
            nodes.append(top_point)
            top_exit = top_point + double_standoff * top_dir
            nodes.append(top_exit)

            if failure_track_id == bottom["track_id"]:
                # Still execute the real U-turn, then collide while entering bottom.
                turn_far_top = top_exit + turnaround_extra * top_dir + turn_side * turnaround_lateral * top_right
                nodes.append(turn_far_top)
                bottom_collision = _sample_collision_point(rng, bottom, failure_cfg)
                bottom_pre = bottom_collision - double_standoff * bottom_dir
                turn_far_bottom = bottom_pre + turnaround_extra * top_dir + turn_side * turnaround_lateral * top_right
                nodes.append(turn_far_bottom)
                nodes.append(bottom_pre)
                active_route_gates.append(bottom)
                gate_node_indices.append(len(nodes))
                nodes.append(bottom_collision)

                impact_cfg = failure_cfg.get("impact", {})
                penetration = float(impact_cfg.get("penetration_m", 0.08))
                drop_distance = float(impact_cfg.get("drop_distance_m", 1.25))
                drop_forward = float(impact_cfg.get("drop_forward_m", 0.25))
                min_final_z = float(impact_cfg.get("min_final_z_m", 0.12))
                drop_to_ground = bool(impact_cfg.get("drop_to_ground", True))
                nodes.append(bottom_collision + penetration * bottom_dir - 0.18 * WORLD_UP)
                final = bottom_collision + drop_forward * bottom_dir
                final[2] = min_final_z if drop_to_ground else max(min_final_z, bottom_collision[2] - drop_distance)
                nodes.append(final)
                stopped_for_failure = True
                break

            # Successful U-turn and bottom pass.
            turn_far_top = top_exit + turnaround_extra * top_dir + turn_side * turnaround_lateral * top_right
            nodes.append(turn_far_top)
            bottom_pre = bottom_point - double_standoff * bottom_dir
            turn_far_bottom = bottom_pre + turnaround_extra * top_dir + turn_side * turnaround_lateral * top_right
            nodes.append(turn_far_bottom)
            nodes.append(bottom_pre)
            active_route_gates.append(bottom)
            gate_node_indices.append(len(nodes))
            nodes.append(bottom_point)
            nodes.append(bottom_point + double_standoff * bottom_dir)

            i += 2
            continue

        if failure_track_id == track_id:
            active_route_gates.append(g)
            if track_id == "gate_06":
                _append_gate6_hairpin_failure_nodes(
                    rng, nodes, gate_node_indices, g, cfg, failure_cfg
                )
            else:
                _append_failure_nodes(
                    rng, nodes, gate_node_indices, g, cfg, failure_cfg
                )
            stopped_for_failure = True
            break

        # Normal successful gate. Gate 6 is special: Gate 5 is on the front side
        # of Gate 6, so we must first go around the map-top end of the gate, get
        # behind it, turn back, and only then fly through the opening toward Gate 7.
        point = crossing_points[track_id]
        active_route_gates.append(g)
        if track_id == "gate_06":
            _append_gate6_hairpin_success_nodes(
                nodes, gate_node_indices, g, point, cfg
            )
        else:
            _append_gate_success_nodes(
                nodes, gate_node_indices, g, point, standoff, standoff
            )
        i += 1

    if not stopped_for_failure:
        last_gate = route_gates[-1]
        last_dir = _rotation_deg_to_flight_dir(float(last_gate["course_heading_deg"]))
        nodes.append(crossing_points[last_gate["track_id"]] + exit_distance * last_dir)

    nodes_arr = np.asarray(nodes, dtype=np.float64)
    if len(nodes_arr) < 4:
        raise RuntimeError("Not enough path nodes to build a trajectory")

    chord = np.linalg.norm(np.diff(nodes_arr, axis=0), axis=1)
    u_nodes = np.concatenate([[0.0], np.cumsum(np.maximum(chord, 1e-4))])
    u_nodes /= u_nodes[-1]

    splines = [
        CubicSpline(u_nodes, nodes_arr[:, ax], bc_type="natural")
        for ax in range(3)
    ]

    dense_n = max(int(cfg.get("path_dense_samples", 6144)), 1024)
    u_dense = np.linspace(0.0, 1.0, dense_n)
    p_dense = np.stack([s(u_dense) for s in splines], axis=1)
    dp_dense = np.stack([s(u_dense, 1) for s in splines], axis=1)

    ds = np.linalg.norm(np.diff(p_dense, axis=0), axis=1)
    s_dense = np.concatenate([[0.0], np.cumsum(ds)])
    total_length = float(s_dense[-1])

    if len(gate_node_indices) != len(active_route_gates):
        raise RuntimeError(
            f"Internal path error: {len(gate_node_indices)} gate nodes for "
            f"{len(active_route_gates)} active route gates"
        )

    gate_s = []
    for g, node_idx in zip(active_route_gates, gate_node_indices):
        u_gate = float(u_nodes[node_idx])
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
        "route_gates": route_gates,
        "active_route_gates": active_route_gates,
        "intentional_failure": failure_plan is not None,
        "failure_target_track_id": failure_track_id,
    }


def _interp_path(path: dict, s_query: float):
    s_dense = path["s_dense"]
    u_dense = path["u_dense"]
    p_dense = path["p_dense"]
    dp_dense = path["dp_dense"]

    sq = float(np.clip(s_query, 0.0, s_dense[-1]))
    u = float(np.interp(sq, s_dense, u_dense))

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
    profile["accel_limit_mps2"] = float(rng.uniform(*profile["accel_limit_mps2_range"]))
    profile["speed_variation_fraction"] = float(rng.uniform(*profile["speed_variation_fraction_range"]))
    profile["orientation_jitter_deg"] = float(rng.uniform(*profile["orientation_jitter_deg_range"]))
    profile["accel_limit_variation_fraction"] = float(
        rng.uniform(*profile.get("accel_limit_variation_fraction_range", [0.10, 0.35]))
    )
    return profile


def _target_speed_function(rng: np.random.Generator, path: dict, profile: dict, motion_cfg: dict):
    s_dense = path["s_dense"]
    total = max(float(s_dense[-1]), 1e-6)
    sn = s_dense / total

    base = float(profile["cruise_speed_mps"])
    amp = float(profile["speed_variation_fraction"])

    c1 = float(rng.uniform(0.6, 1.8))
    c2 = float(rng.uniform(1.8, 4.5))
    p1 = float(rng.uniform(0.0, 2.0 * math.pi))
    p2 = float(rng.uniform(0.0, 2.0 * math.pi))

    v = base * (
        1.0
        + 0.70 * amp * np.sin(2.0 * math.pi * c1 * sn + p1)
        + 0.30 * amp * np.sin(2.0 * math.pi * c2 * sn + p2)
    )

    event_range = motion_cfg.get("speed_event_count_range", [2, 7])
    lo, hi = int(event_range[0]), int(event_range[1])
    events = int(rng.integers(lo, hi + 1))
    magnitude_range = motion_cfg.get("speed_event_fraction_range", [-0.45, 0.55])
    width_range = motion_cfg.get("speed_event_width_fraction_range", [0.025, 0.12])
    event_meta = []

    for _ in range(events):
        center = float(rng.uniform(0.08, 0.92))
        width = float(rng.uniform(*width_range))
        magnitude = float(rng.uniform(*magnitude_range))
        if abs(magnitude) < 0.05:
            magnitude = 0.05 if rng.random() < 0.5 else -0.05

        bump = np.exp(-0.5 * ((sn - center) / width) ** 2)
        v *= (1.0 + magnitude * bump)
        event_meta.append({
            "center_fraction": center,
            "width_fraction": width,
            "fractional_speed_change": magnitude,
        })

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


def generate_camera_trajectory(
    rng: np.random.Generator,
    gates: list[dict],
    trajectory_cfg: dict,
    motion_cfg: dict,
    fps: float,
    failure_plan: dict | None = None,
    failure_cfg: dict | None = None,
):
    """Time-sample the fixed route, including optional intentional failure."""
    failure_cfg = failure_cfg or {}
    path = _build_spatial_path(
        rng,
        gates,
        trajectory_cfg,
        failure_plan=failure_plan,
        failure_cfg=failure_cfg,
    )
    active_route_gates = path["active_route_gates"]
    profile = _sample_motion_profile(rng, motion_cfg)
    target_dense, speed_events = _target_speed_function(rng, path, profile, motion_cfg)

    dt = 1.0 / float(fps)
    total = float(path["total_length_m"])
    s = 0.0

    start_frac = float(profile.get("start_speed_fraction", 0.75))
    v = max(float(profile.get("min_speed_mps", 0.5)), float(profile["cruise_speed_mps"]) * start_frac)
    accel_limit = float(profile["accel_limit_mps2"])
    accel_var = float(profile.get("accel_limit_variation_fraction", 0.2))
    accel_phase = float(rng.uniform(0.0, 2.0 * math.pi))
    accel_cycles = float(rng.uniform(0.7, 2.8))

    samples = []
    max_frames = int(motion_cfg.get("max_frames_per_sequence", 6500))
    fi = 0

    while s < total - 1e-5 and fi < max_frames:
        target_v = float(np.interp(s, path["s_dense"], target_dense))
        progress = s / max(total, 1e-9)
        dynamic_accel_limit = accel_limit * (
            1.0 + accel_var * math.sin(2.0 * math.pi * accel_cycles * progress + accel_phase)
        )
        dynamic_accel_limit = max(0.25, dynamic_accel_limit)

        dv = np.clip(target_v - v, -dynamic_accel_limit * dt, dynamic_accel_limit * dt)
        v = max(0.20, v + float(dv))

        pos, tangent = _interp_path(path, s)
        samples.append({
            "path_s_m": float(s),
            "position_world_m": pos,
            "tangent_world": tangent,
            "speed_mps": float(v),
            "target_speed_mps": target_v,
            "dynamic_accel_limit_mps2": float(dynamic_accel_limit),
        })

        s = min(total, s + v * dt)
        fi += 1

    truncated_by_max_frames = bool(fi >= max_frames and s < total - 1e-5)

    if not truncated_by_max_frames and (not samples or samples[-1]["path_s_m"] < total - 1e-4):
        pos, tangent = _interp_path(path, total)
        samples.append({
            "path_s_m": total,
            "position_world_m": pos,
            "tangent_world": tangent,
            "speed_mps": float(v),
            "target_speed_mps": float(np.interp(total, path["s_dense"], target_dense)),
            "dynamic_accel_limit_mps2": float(accel_limit),
        })

    phase_y = float(rng.uniform(0.0, 2.0 * math.pi))
    phase_z = float(rng.uniform(0.0, 2.0 * math.pi))
    gaze_cycles = float(rng.uniform(0.6, 2.2))

    jitter_deg = float(profile["orientation_jitter_deg"])
    lookahead = float(trajectory_cfg.get("camera_lookahead_m", 4.0))
    gate_s = path["gate_s_m"]
    pass_margin = float(trajectory_cfg.get("gate_pass_margin_m", 0.35))

    for st in samples:
        s_now = st["path_s_m"]
        pos = st["position_world_m"]
        tangent = st["tangent_world"]
        tnorm = s_now / max(total, 1e-9)

        ahead_idx = np.where(gate_s >= s_now - pass_margin)[0]
        current_idx = int(ahead_idx[0]) if len(ahead_idx) else None
        if current_idx is not None:
            st["current_target_route_index"] = active_route_gates[current_idx]["route_order_index"]
        else:
            st["current_target_route_index"] = None

        # Smooth aim: look ahead on the continuous spline, not directly at a
        # gate center. This removes the small orientation jump at gate changes.
        look_s = min(s_now + lookahead, total)
        path_aim, _ = _interp_path(path, look_s)
        aim = path_aim

        horizontal = tangent.copy()
        horizontal[2] = 0.0
        if np.linalg.norm(horizontal) < 1e-8:
            horizontal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        horizontal /= max(np.linalg.norm(horizontal), 1e-9)
        lateral = np.array([-horizontal[1], horizontal[0], 0.0], dtype=np.float64)

        yaw_offset = math.tan(
            math.radians(
                jitter_deg * 0.55 * math.sin(2.0 * math.pi * gaze_cycles * tnorm + phase_y)
            )
        ) * lookahead
        pitch_offset = math.tan(
            math.radians(
                jitter_deg * 0.35 * math.sin(2.0 * math.pi * 0.73 * gaze_cycles * tnorm + phase_z)
            )
        ) * lookahead

        aim = aim + lateral * yaw_offset + np.array([0.0, 0.0, pitch_offset], dtype=np.float64)
        T_wc = look_at_T_world_camera(pos, aim)
        st["T_world_camera"] = T_wc
        st["velocity_world_mps"] = tangent * st["speed_mps"]

    for i, st in enumerate(samples):
        if len(samples) == 1:
            acc = np.zeros(3, dtype=np.float64)
            omega = np.zeros(3, dtype=np.float64)
        elif i == 0:
            acc = (samples[1]["velocity_world_mps"] - st["velocity_world_mps"]) / dt
            omega = _angular_velocity_local(st["T_world_camera"], samples[1]["T_world_camera"], dt)
        else:
            acc = (st["velocity_world_mps"] - samples[i - 1]["velocity_world_mps"]) / dt
            omega = _angular_velocity_local(samples[i - 1]["T_world_camera"], st["T_world_camera"], dt)

        st["acceleration_world_mps2"] = acc
        st["angular_velocity_camera_radps"] = omega
        st["frame_index"] = i
        st["time_s"] = i * dt

    intentional_failure = failure_plan is not None
    completed_course = (not intentional_failure) and (not truncated_by_max_frames)
    failure_target = failure_plan.get("target_track_id") if failure_plan else None
    failure_route_index = None
    if failure_target is not None:
        for g in gates:
            if g["track_id"] == failure_target:
                failure_route_index = int(g["route_order_index"])
                break

    outcome = {
        "status": "failure" if intentional_failure else "success",
        "intentional_failure": bool(intentional_failure),
        "failure_type": "gate_collision_drop" if intentional_failure else None,
        "failure_target_track_id": failure_target,
        "failure_target_route_index": failure_route_index,
    }

    meta = {
        "motion_profile": profile,
        "speed_events": speed_events,
        "path_length_m": total,
        "num_frames": len(samples),
        "duration_s": (len(samples) - 1) * dt if samples else 0.0,
        "speed_min_mps": float(min(smp["speed_mps"] for smp in samples)),
        "speed_max_mps": float(max(smp["speed_mps"] for smp in samples)),
        "speed_mean_mps": float(np.mean([smp["speed_mps"] for smp in samples])),
        "acceleration_mean_mps2": float(np.mean([np.linalg.norm(smp["acceleration_world_mps2"]) for smp in samples])),
        "acceleration_max_mps2": float(np.max([np.linalg.norm(smp["acceleration_world_mps2"]) for smp in samples])),
        "course_style": "fixed_track_map",
        "completed_course": completed_course,
        "truncated_by_max_frames": truncated_by_max_frames,
        "traversed_length_m": float(samples[-1]["path_s_m"]) if samples else 0.0,
        "track_source": "Drone_Race_Track_Gate_Coordinates_with_doublegate.pdf",
        "outcome": outcome,
    }

    return samples, meta
