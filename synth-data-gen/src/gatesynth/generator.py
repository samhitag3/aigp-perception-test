from __future__ import annotations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from .augment import sample_sequence_noise, apply_photometric, apply_photometric_batch
from .geometry import DEPTH, OUTER_W, OUTER_H, INNER_W, INNER_H
from .texture import load_gate_skin
from .trajectory import generate_course, generate_camera_trajectory
from .render import render_frame, procedural_background, load_background_frame
from .writer import ensure_sequence_dirs, save_rgb, save_mask, json_dump, write_jsonl, write_checksums


def _choose_background(rng, cfg, width, height):
    d = cfg["render"].get("backgrounds_dir")
    if d:
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            paths += list(Path(d).glob(ext))
        if paths and rng.random() > cfg["render"].get("procedural_background_probability", 0.5):
            p = paths[int(rng.integers(0, len(paths)))]
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is not None:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), str(p)
    return None, "procedural"


def _dataset_json(cfg):
    W, H, fx, fy, cx, cy, K, hfov, vfov = _fixed_camera(cfg)
    return {
        "schema_name": "uav_gate_perception_dataset",
        "schema_version": cfg["schema_version"],
        "dataset_id": cfg["dataset_id"],
        "coordinate_systems": {
            "world": {"handedness": "right", "x_axis": "forward_reference", "y_axis": "left", "z_axis": "up", "units": "meters"},
            "drone_body": {"convention": "camera_aligned_for_synthetic_generator", "handedness": "right"},
            "camera_optical": {"x_axis": "right", "y_axis": "down", "z_axis": "forward", "handedness": "right"},
            "gate_local": {"x_axis": "right_when_viewed_from_approach_side", "y_axis": "down_when_viewed_from_approach_side", "z_axis": "through_gate_from_approach_side", "handedness": "right"},
        },
        "transform_convention": {"name": "T_A_B", "definition": "T_A_B transforms a homogeneous point expressed in coordinate frame B into coordinate frame A", "matrix_layout": "row_major", "quaternion_order": "xyzw", "translation_units": "meters"},
        "image_coordinate_system": {"origin": "top_left", "x_direction": "right", "y_direction": "down", "pixel_coordinate_type": "float", "top_left_pixel_center": [0.0, 0.0], "in_frame_rule": "0 <= x < image_width and 0 <= y < image_height", "normalized_x": "x / (image_width - 1)", "normalized_y": "y / (image_height - 1)"},
        "gate_keypoint_order": ["outer_tl", "outer_tr", "outer_br", "outer_bl", "inner_tl", "inner_tr", "inner_br", "inner_bl"],
        "mask_encoding": {"format": "PNG", "dtype": "uint16", "channels": 1, "background_value": 0, "meaning": "Each nonzero integer is a frame-local gate mask_id"},
        "rgb_encoding": {"format": "JPEG", "color_space": "sRGB", "channels": 3, "dtype": "uint8"},
        "depth_encoding": {"format": "PNG", "dtype": "uint16", "units": "millimeters", "invalid_value": 0, "meaning": "camera optical Z depth, not Euclidean range"},
        "required_modalities": {"rgb": True, "instance_mask": True, "frame_annotations": True, "camera_intrinsics": True, "camera_pose": True, "drone_pose": True, "gate_pose": True, "gate_geometry": True, "projected_keypoints": True, "keypoint_visibility": True},
        "optional_modalities": {"dense_depth": False, "optical_flow": False, "surface_normals": False, "imu": False, "control_actions": False},
        "splitting": {"unit": "sequence", "train_fraction": cfg["split"]["train"], "validation_fraction": cfg["split"]["validation"], "test_fraction": cfg["split"]["test"], "frames_from_same_sequence_may_cross_splits": False},
        "camera_spec": {
            "model": "pinhole",
            "sensor_shutter": "rolling",
            "width_px": W,
            "height_px": H,
            "fps": float(cfg["sequences"]["fps"]),
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "K": K.tolist(),
            "HFoV_deg": hfov,
            "VFoV_deg": vfov,
            "distortion_model": "none",
            "rolling_shutter_readout_time_s": cfg["camera"]
                .get("shutter", {})
                .get("readout_time_s")
        },
        "camera_calibration_policy": {"fixed": True, "randomize_resolution": False, "randomize_focal_length": False, "randomize_principal_point": False, "randomize_distortion": False},
    }


def _gate_geometry_json(cfg):
    z = -DEPTH / 2
    return {
        "schema_version": cfg["schema_version"],
        "gate_types": {
            "standard_gate": {
                "outer_width_m": OUTER_W,
                "outer_height_m": OUTER_H,
                "inner_width_m": INNER_W,
                "inner_height_m": INNER_H,
                "depth_m": DEPTH,
                "keypoint_reference_surface": "approach_face",
                "keypoints_gate_frame_m": {
                    "outer_tl": [-OUTER_W / 2, -OUTER_H / 2, z],
                    "outer_tr": [OUTER_W / 2, -OUTER_H / 2, z],
                    "outer_br": [OUTER_W / 2, OUTER_H / 2, z],
                    "outer_bl": [-OUTER_W / 2, OUTER_H / 2, z],
                    "inner_tl": [-INNER_W / 2, -INNER_H / 2, z],
                    "inner_tr": [INNER_W / 2, -INNER_H / 2, z],
                    "inner_br": [INNER_W / 2, INNER_H / 2, z],
                    "inner_bl": [-INNER_W / 2, INNER_H / 2, z],
                },
            }
        },
    }


def _make_split_map(seq_ids: list[str], cfg: dict, base_seed: int):
    split_rng = np.random.default_rng(base_seed + 99173)
    shuffled = seq_ids.copy()
    split_rng.shuffle(shuffled)
    nseq = len(seq_ids)
    ntr = int(round(nseq * cfg["split"]["train"]))
    nv = int(round(nseq * cfg["split"]["validation"]))
    split_map = {s: "train" for s in shuffled[:ntr]}
    split_map.update({s: "validation" for s in shuffled[ntr:ntr + nv]})
    split_map.update({s: "test" for s in shuffled[ntr + nv:]})
    return split_map


def _fixed_camera(cfg: dict):
    W, H = int(cfg["image"]["width"]), int(cfg["image"]["height"])
    cam = cfg["camera"]
    expected_w, expected_h = int(cam["width"]), int(cam["height"])
    if (W, H) != (expected_w, expected_h):
        raise ValueError(f"image resolution {W}x{H} does not match camera resolution {expected_w}x{expected_h}")
    fx, fy = float(cam["fx"]), float(cam["fy"])
    cx, cy = float(cam["cx"]), float(cam["cy"])
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    hfov = math.degrees(2.0 * math.atan(W / (2.0 * fx)))
    vfov = math.degrees(2.0 * math.atan(H / (2.0 * fy)))
    return W, H, fx, fy, cx, cy, K, hfov, vfov


def _route_targets(gates, state):
    idx = state.get("current_target_route_index")
    if idx is None or idx < 0 or idx >= len(gates):
        return None, None
    current = gates[int(idx)]
    nextg = gates[int(idx) + 1] if int(idx) + 1 < len(gates) else None
    return current, nextg


def _frame_record(cfg, seq_id, fi, fps, rgb_rel, mask_rel, W, H, fx, fy, cx, cy, K, state, current, nextg, gate_records, noise, motion_meta):
    pos = state["position_world_m"]
    vel = state["velocity_world_mps"]
    acc = state["acceleration_world_mps2"]
    omega = state["angular_velocity_camera_radps"]
    T_wc = state["T_world_camera"]
    R_wc = T_wc[:3, :3]
    q = Rotation.from_matrix(R_wc).as_quat()
    vel_local = R_wc.T @ vel
    acc_local = R_wc.T @ acc
    speed = float(state["speed_mps"])
    return {
        "schema_version": cfg["schema_version"],
        "sequence_id": seq_id,
        "frame_index": fi,
        "frame_id": f"{seq_id}_frame_{fi:06d}",
        "timestamp_ns": int(round(fi / fps * 1e9)),
        "sim_time_s": fi / fps,
        "files": {"rgb": rgb_rel, "instance_mask": mask_rel, "depth_mm": None, "optical_flow": None, "surface_normals": None},
        "image": {"width_px": W, "height_px": H},
        "camera": {
            "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "K": K.tolist()},
            "T_world_camera": T_wc.tolist(),
            "position_world_m": [float(v) for v in pos],
            "quaternion_world_xyzw": [float(v) for v in q],
        },
        "drone": {
            "T_world_body": T_wc.tolist(),
            "position_world_m": [float(v) for v in pos],
            "quaternion_world_xyzw": [float(v) for v in q],
            "linear_velocity_world_mps": [float(v) for v in vel],
            "linear_velocity_body_mps": [float(v) for v in vel_local],
            "angular_velocity_body_radps": [float(v) for v in omega],
            "linear_acceleration_body_mps2": [float(v) for v in acc_local],
        },
        "route": {
            "current_target_track_id": current["track_id"] if current else None,
            "next_target_track_id": nextg["track_id"] if nextg else None,
            "current_target_route_index": current["route_order_index"] if current else None,
            "path_progress_m": float(state["path_s_m"]),
            "path_length_m": float(motion_meta["path_length_m"]),
        },
        "gates": gate_records,
        "frame_conditions": {
            "gate_count_visible": int(sum(1 for g in gate_records if g["visibility"]["has_visible_pixels"])),
            "gate_count_annotated": len(gate_records),
            "drone_speed_mps": speed,
            "target_speed_mps": float(state["target_speed_mps"]),
            "drone_acceleration_mps2": float(np.linalg.norm(acc)),
            "motion_profile": motion_meta["motion_profile"]["name"],
            "course_style": motion_meta.get("course_style"),
            "render_randomization": {
                "tier": noise["tier"],
                "brightness_factor": noise["brightness"],
                "contrast_factor": noise["contrast"],
                "gamma": noise["gamma"],
                "motion_blur_strength": noise["motion_blur_px"],
                "noise_strength": noise["gaussian_noise_std"],
                "speckle_std": noise["speckle_std"],
                "extra": {},
            },
        },
    }


def _flush_gpu_chunk(cfg, seqdir, seq_id, fps, W, H, fx, fy, cx, cy, K, gates, frames, noise, motion_meta,
                     pending, rgb_quality, rng_seed, device):
    if not pending:
        return []
    frame_indices = [item[0] for item in pending]
    clean_imgs = [item[2] for item in pending]
    noisy_imgs = apply_photometric_batch(clean_imgs, noise, frame_indices, frames, rng_seed=rng_seed, device=device)
    records = []
    for (fi, state, clean, mask, gate_records), noisy in zip(pending, noisy_imgs):
        rgb_rel = f"rgb/frame_{fi:06d}.jpg"
        mask_rel = f"instance_masks/frame_{fi:06d}.png"
        save_rgb(seqdir / rgb_rel, noisy, rgb_quality)
        save_mask(seqdir / mask_rel, mask)
        current, nextg = _route_targets(gates, state)
        for gr in gate_records:
            gr["is_current_target"] = bool(current and gr["track_id"] == current["track_id"])
        records.append(_frame_record(cfg, seq_id, fi, fps, rgb_rel, mask_rel, W, H, fx, fy, cx, cy, K, state, current, nextg, gate_records, noise, motion_meta))
    pending.clear()
    return records


def _generate_sequence(task: dict):
    cfg = task["cfg"]
    seq_id = task["seq_id"]
    seq_index = int(task["seq_index"])
    split_name = task["split_name"]
    gate_skin_path = task["gate_skin_path"]
    root = Path(task["root"])

    accel = cfg.get("acceleration", {})
    photometric_backend = str(accel.get("photometric_backend", "cpu")).lower()
    photometric_device = str(accel.get("device", "cpu"))
    frame_batch_size = int(accel.get("frame_batch_size", 16))

    base_seed = int(cfg["seed"])
    sequence_seed = base_seed + seq_index * 100003
    rng = np.random.default_rng(sequence_seed)
    W, H, fx, fy, cx, cy, K, hfov, vfov = _fixed_camera(cfg)
    fps = float(cfg["sequences"]["fps"])

    skin = load_gate_skin(gate_skin_path)
    gates = generate_course(rng, cfg["course"])
    traj, motion_meta = generate_camera_trajectory(rng, gates, cfg["trajectory"], cfg["motion"], fps)
    frames = len(traj)
    noise = sample_sequence_noise(rng, cfg["noise"])
    bg, bg_id = _choose_background(rng, cfg, W, H)

    seqdir = ensure_sequence_dirs(root, seq_id)
    seq_meta = {
        "schema_version": cfg["schema_version"],
        "sequence_id": seq_id,
        "split": split_name,
        "num_frames": frames,
        "simulator": {"name": "synthetic_projective_renderer", "version": "0.3.0-diverse-motion", "simulation_seed": int(sequence_seed)},
        "timing": {"nominal_fps": fps, "nominal_dt_s": 1.0 / fps, "duration_s": float(motion_meta["duration_s"])},
        "environment": {"environment_id": bg_id, "scene_id": f"scene_{seq_index:06d}"},
        "camera": {
            "camera_id": "front_camera",
            "sensor": {
                "module": "Arducam 12.3 MP HQ Camera Module",
                "lens_mount": "M12",
                "shutter_type": "rolling"
            },
            "width_px": W,
            "height_px": H,
            "model": "pinhole",
            "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "K": K.tolist(), "HFoV_deg": hfov, "VFoV_deg": vfov},
            "distortion": {"model": "none", "coefficients_order": ["k1", "k2", "p1", "p2", "k3"], "coefficients": [0, 0, 0, 0, 0]},
            "calibration_randomized": False,
            "T_body_camera": np.eye(4).tolist(),
        },
        "course": {
            "style": motion_meta.get("course_style"),
            "gate_tracks": [{
                "track_id": g["track_id"],
                "gate_type_id": "standard_gate",
                "route_order_index": g["route_order_index"],
                "T_world_gate": g["T_world_gate"].tolist(),
                "position_world_m": [float(v) for v in g["position_world"]],
                "course_heading_deg": float(g["course_heading_deg"]),
                "turn_from_previous_deg": float(g["turn_from_previous_deg"]),
                "gate_yaw_offset_deg": float(g["gate_yaw_offset_deg"]),
                "gate_pitch_deg": float(g["gate_pitch_deg"]),
                "gate_roll_deg": float(g["gate_roll_deg"]),
                "path_s_m": float(g.get("path_s_m", 0.0)),
            } for g in gates],
        },
        "motion": motion_meta,
        "domain_randomization": {"enabled": True, "noise_tier": noise["tier"], "sequence_fixed_parameters": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in noise.items()}, "background_id": bg_id},
    }
    json_dump(seqdir / "sequence.json", seq_meta)

    use_gpu_batch = photometric_backend in {"torch", "gpu", "cuda", "auto"}
    if photometric_backend == "auto":
        try:
            import torch
            use_gpu_batch = bool(torch.cuda.is_available())
            if use_gpu_batch and photometric_device == "cpu":
                photometric_device = "cuda"
        except Exception:
            use_gpu_batch = False
    if photometric_device.startswith("cuda") and os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        use_gpu_batch = False

    records = []
    pending = []
    for fi, state in enumerate(traj):
        frng_seed = sequence_seed + fi * 97 + 17
        frng = np.random.default_rng(frng_seed)
        b = load_background_frame(bg, W, H, fi, frames)
        if b is None:
            b = procedural_background(frng, W, H, fi, frames)
        T_wc = state["T_world_camera"]
        clean, mask, gate_records = render_frame(
            b, skin, gates, T_wc, K, W, H,
            cfg["render"]["include_side_faces"],
            cfg["render"]["side_face_brightness"],
        )

        if use_gpu_batch:
            pending.append((fi, state, clean, mask, gate_records))
            if len(pending) >= frame_batch_size:
                records.extend(_flush_gpu_chunk(cfg, seqdir, seq_id, fps, W, H, fx, fy, cx, cy, K, gates, frames, noise, motion_meta, pending, cfg["image"]["jpeg_quality"], rng_seed=frng_seed, device=photometric_device))
        else:
            noisy = apply_photometric(clean, noise, fi, frames, frng)
            rgb_rel = f"rgb/frame_{fi:06d}.jpg"
            mask_rel = f"instance_masks/frame_{fi:06d}.png"
            save_rgb(seqdir / rgb_rel, noisy, cfg["image"]["jpeg_quality"])
            save_mask(seqdir / mask_rel, mask)
            current, nextg = _route_targets(gates, state)
            for gr in gate_records:
                gr["is_current_target"] = bool(current and gr["track_id"] == current["track_id"])
            records.append(_frame_record(cfg, seq_id, fi, fps, rgb_rel, mask_rel, W, H, fx, fy, cx, cy, K, state, current, nextg, gate_records, noise, motion_meta))

    if use_gpu_batch and pending:
        records.extend(_flush_gpu_chunk(cfg, seqdir, seq_id, fps, W, H, fx, fy, cx, cy, K, gates, frames, noise, motion_meta, pending, cfg["image"]["jpeg_quality"], rng_seed=sequence_seed + 99991, device=photometric_device))

    records.sort(key=lambda r: r["frame_index"])
    write_jsonl(seqdir / "frames.jsonl", records)
    return {"sequence_id": seq_id, "num_frames": len(records), "split": split_name, "motion_profile": motion_meta["motion_profile"]["name"], "course_style": motion_meta.get("course_style")}


def generate_dataset(cfg: dict, gate_skin_path: str, output_root: str | None = None):
    root = Path(output_root or cfg["output_root"]).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "splits").mkdir(exist_ok=True)
    (root / "sequences").mkdir(exist_ok=True)

    # Fail fast on calibration mismatch before generating anything.
    _fixed_camera(cfg)
    json_dump(root / "dataset.json", _dataset_json(cfg))
    json_dump(root / "gate_geometry.json", _gate_geometry_json(cfg))

    nseq = int(cfg["sequences"]["count"])
    base_seed = int(cfg["seed"])
    seq_ids = [f"seq_{i:06d}" for i in range(nseq)]
    split_map = _make_split_map(seq_ids, cfg, base_seed)
    for name in ("train", "validation", "test"):
        ids = [s for s in seq_ids if split_map[s] == name]
        (root / "splits" / f"{name}_sequences.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")

    gen_cfg = cfg.get("generation", {})
    num_workers = int(gen_cfg.get("num_workers", 1))
    write_checksums_flag = bool(gen_cfg.get("write_checksums", True))

    tasks = [{
        "cfg": cfg,
        "seq_id": seq_id,
        "seq_index": i,
        "split_name": split_map[seq_id],
        "gate_skin_path": str(gate_skin_path),
        "root": str(root),
    } for i, seq_id in enumerate(seq_ids)]

    if num_workers <= 1:
        for task in tqdm(tasks, desc="Sequences"):
            _generate_sequence(task)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(_generate_sequence, t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Sequences"):
                # Surface worker errors instead of silently continuing.
                fut.result()

    if write_checksums_flag:
        write_checksums(root)
    return root
