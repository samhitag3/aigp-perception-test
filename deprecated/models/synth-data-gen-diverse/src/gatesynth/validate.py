from __future__ import annotations
from pathlib import Path
import json
import cv2
import numpy as np

REQ_KP = ["outer_tl", "outer_tr", "outer_br", "outer_bl", "inner_tl", "inner_tr", "inner_br", "inner_bl"]


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_dataset(root: str | Path) -> dict:
    root = Path(root)
    errors, warnings = [], []
    nframes = 0
    nseq = 0
    profile_counts = {}
    course_style_counts = {}

    dataset_path = root / "dataset.json"
    if not dataset_path.exists():
        return {"valid": False, "sequences": 0, "frames": 0, "errors": [f"missing {dataset_path}"], "warnings": []}
    dataset = _load_json(dataset_path)
    camera_spec = dataset.get("camera_spec", {})
    expected_W = camera_spec.get("width_px")
    expected_H = camera_spec.get("height_px")
    expected_K = np.asarray(camera_spec.get("K"), dtype=float) if camera_spec.get("K") is not None else None

    splits = {}
    for s in ("train", "validation", "test"):
        p = root / "splits" / f"{s}_sequences.txt"
        if not p.exists():
            errors.append(f"missing split file {p}")
            splits[s] = set()
        else:
            splits[s] = set(x.strip() for x in p.read_text().splitlines() if x.strip())
    if splits["train"] & splits["validation"] or splits["train"] & splits["test"] or splits["validation"] & splits["test"]:
        errors.append("sequence split overlap detected")

    all_seq_ids = set()
    for seqdir in sorted((root / "sequences").glob("seq_*")):
        nseq += 1
        all_seq_ids.add(seqdir.name)
        sp = seqdir / "sequence.json"
        fp = seqdir / "frames.jsonl"
        if not sp.exists():
            errors.append(f"missing {sp}")
            continue
        if not fp.exists():
            errors.append(f"missing {fp}")
            continue

        sm = _load_json(sp)
        cam = sm.get("camera", {})
        if expected_W is not None and int(cam.get("width_px", -1)) != int(expected_W):
            errors.append(f"{seqdir.name}: width differs from dataset camera spec")
        if expected_H is not None and int(cam.get("height_px", -1)) != int(expected_H):
            errors.append(f"{seqdir.name}: height differs from dataset camera spec")
        Kseq = np.asarray(cam.get("intrinsics", {}).get("K"), dtype=float)
        if expected_K is not None and (Kseq.shape != (3, 3) or not np.allclose(Kseq, expected_K, atol=1e-9)):
            errors.append(f"{seqdir.name}: K differs from fixed dataset camera spec")
        if cam.get("distortion", {}).get("model") != "none":
            errors.append(f"{seqdir.name}: distortion must be disabled")
        if cam.get("calibration_randomized") is not False:
            warnings.append(f"{seqdir.name}: calibration_randomized is not explicitly false")

        motion = sm.get("motion", {})
        p_name = motion.get("motion_profile", {}).get("name", "unknown")
        c_name = motion.get("course_style", "unknown")
        profile_counts[p_name] = profile_counts.get(p_name, 0) + 1
        course_style_counts[c_name] = course_style_counts.get(c_name, 0) + 1

        gate_tracks = sm.get("course", {}).get("gate_tracks", [])
        track_ids = {g["track_id"] for g in gate_tracks}
        route_indices = [g.get("route_order_index") for g in gate_tracks]
        if route_indices != sorted(route_indices):
            errors.append(f"{seqdir.name}: route_order_index not sorted")
        gate_s = [g.get("path_s_m") for g in gate_tracks if g.get("path_s_m") is not None]
        if len(gate_s) > 1 and np.any(np.diff(gate_s) <= 0):
            errors.append(f"{seqdir.name}: gate path_s_m is not strictly increasing")

        last_idx = -1
        last_ts = -1
        last_path_s = -np.inf
        seq_frames = 0
        with open(fp, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                if not line.strip():
                    continue
                r = json.loads(line)
                nframes += 1
                seq_frames += 1
                if r["frame_index"] <= last_idx:
                    errors.append(f"{fp}:{ln} non-increasing frame_index")
                if r["timestamp_ns"] <= last_ts and last_ts >= 0:
                    errors.append(f"{fp}:{ln} non-increasing timestamp")
                last_idx = r["frame_index"]
                last_ts = r["timestamp_ns"]

                path_s = float(r.get("route", {}).get("path_progress_m", -1.0))
                if path_s + 1e-8 < last_path_s:
                    errors.append(f"{fp}:{ln} path_progress_m moved backwards")
                last_path_s = path_s

                rgbp = seqdir / r["files"]["rgb"]
                mp = seqdir / r["files"]["instance_mask"]
                if not rgbp.exists():
                    errors.append(f"missing {rgbp}")
                    continue
                if not mp.exists():
                    errors.append(f"missing {mp}")
                    continue
                rgb = cv2.imread(str(rgbp))
                mask = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
                if rgb is None or mask is None:
                    errors.append(f"unreadable files for {r['frame_id']}")
                    continue
                if rgb.shape[:2] != mask.shape[:2]:
                    errors.append(f"shape mismatch {r['frame_id']}")
                if expected_H is not None and rgb.shape[0] != int(expected_H):
                    errors.append(f"wrong RGB height {r['frame_id']}")
                if expected_W is not None and rgb.shape[1] != int(expected_W):
                    errors.append(f"wrong RGB width {r['frame_id']}")

                Kframe = np.asarray(r.get("camera", {}).get("intrinsics", {}).get("K"), dtype=float)
                if expected_K is not None and (Kframe.shape != (3, 3) or not np.allclose(Kframe, expected_K, atol=1e-9)):
                    errors.append(f"frame K mismatch {r['frame_id']}")

                speed = float(r.get("frame_conditions", {}).get("drone_speed_mps", np.nan))
                if not np.isfinite(speed) or speed < 0:
                    errors.append(f"invalid speed {r['frame_id']}")

                ids = set(int(x) for x in np.unique(mask) if int(x) != 0)
                anno = set(int(g["mask_id"]) for g in r["gates"] if g["mask_id"] is not None)
                if not ids.issubset(anno):
                    errors.append(f"unknown mask id(s) {ids - anno} in {r['frame_id']}")

                for g in r["gates"]:
                    if g["track_id"] not in track_ids:
                        errors.append(f"unknown track_id {g['track_id']} in {r['frame_id']}")
                    if set(g["keypoints_2d"].keys()) != set(REQ_KP):
                        errors.append(f"bad keypoint keys {r['frame_id']} {g['track_id']}")
                    T = np.asarray(g["pose"]["T_camera_gate"], float)
                    if T.shape != (4, 4) or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6):
                        errors.append(f"bad transform {r['frame_id']} {g['track_id']}")
                    if g["mask_id"] is not None:
                        area = int(np.sum(mask == g["mask_id"]))
                        if area != int(g["visibility"]["visible_pixel_area"]):
                            errors.append(f"visible area mismatch {r['frame_id']} {g['track_id']}")

        if seq_frames != int(sm.get("num_frames", -1)):
            errors.append(f"{seqdir.name}: sequence.json num_frames={sm.get('num_frames')} but frames.jsonl has {seq_frames}")

    declared = splits["train"] | splits["validation"] | splits["test"]
    if declared != all_seq_ids:
        missing = all_seq_ids - declared
        extra = declared - all_seq_ids
        if missing:
            errors.append(f"sequences missing from split manifests: {sorted(missing)[:10]}")
        if extra:
            errors.append(f"split manifests reference missing sequences: {sorted(extra)[:10]}")

    return {
        "valid": len(errors) == 0,
        "sequences": nseq,
        "frames": nframes,
        "motion_profile_counts": profile_counts,
        "course_style_counts": course_style_counts,
        "errors": errors[:200],
        "warnings": warnings[:200],
    }
