#!/usr/bin/env python3
"""Validate canonical UAV gate dataset compatibility with vanilla GatePoseNet."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

KEYS = ("outer_tl", "outer_tr", "outer_br", "outer_bl",
        "inner_tl", "inner_tr", "inner_br", "inner_bl")


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield n, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{n}: {exc}") from exc


def valid_T(x):
    a = np.asarray(x, dtype=np.float64) if x is not None else np.empty((0,))
    if a.shape != (4, 4) or not np.isfinite(a).all():
        return False
    if not np.allclose(a[3], [0, 0, 0, 1], atol=1e-5):
        return False
    R = a[:3, :3]
    return np.allclose(R.T @ R, np.eye(3), atol=2e-3) and np.linalg.det(R) > 0.99


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fx", type=float, default=320.0)
    ap.add_argument("--fy", type=float, default=320.0)
    ap.add_argument("--cx", type=float, default=320.0)
    ap.add_argument("--cy", type=float, default=180.0)
    ap.add_argument("--projection-tol-px", type=float, default=0.5)
    ap.add_argument("--max-sequences", type=int, default=None)
    args = ap.parse_args()

    root = Path(args.dataset_root).expanduser().resolve()
    errors, warnings = [], []

    for required in ["dataset.json", "gate_geometry.json"]:
        if not (root / required).exists():
            errors.append(f"missing {required}")
    if errors:
        print("\n".join(f"ERROR: {e}" for e in errors))
        raise SystemExit(1)

    ds = read_json(root / "dataset.json")
    if ds.get("schema_version") != "1.0.0":
        warnings.append(f"schema_version={ds.get('schema_version')!r}; expected 1.0.0")

    split_paths = {
        "train": root / "splits/train_sequences.txt",
        "validation": root / "splits/validation_sequences.txt",
        "test": root / "splits/test_sequences.txt",
    }
    split_sets = {}
    for name, path in split_paths.items():
        if not path.exists():
            errors.append(f"missing split manifest {path.relative_to(root)}")
            split_sets[name] = set()
            continue
        ids = {x.strip() for x in path.read_text().splitlines()
               if x.strip() and not x.lstrip().startswith("#")}
        split_sets[name] = ids
    names = list(split_sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = split_sets[names[i]] & split_sets[names[j]]
            if overlap:
                errors.append(f"split leakage {names[i]} vs {names[j]}: {sorted(overlap)[:8]}")

    expected_K = np.array([[args.fx, 0, args.cx], [0, args.fy, args.cy], [0, 0, 1]], dtype=float)
    hfov = math.degrees(2 * math.atan(args.width / (2 * args.fx)))
    vfov = math.degrees(2 * math.atan(args.height / (2 * args.fy)))

    sequence_ids = sorted(set().union(*split_sets.values()))
    if args.max_sequences is not None:
        sequence_ids = sequence_ids[:args.max_sequences]

    n_frames = n_masks = n_gates = 0
    for sid in sequence_ids:
        seq_dir = root / "sequences" / sid
        seq_json = seq_dir / "sequence.json"
        frames_jsonl = seq_dir / "frames.jsonl"
        if not seq_json.exists() or not frames_jsonl.exists():
            errors.append(f"{sid}: missing sequence.json/frames.jsonl")
            continue
        seq = read_json(seq_json)
        if seq.get("sequence_id") != sid:
            errors.append(f"{sid}: sequence_id mismatch {seq.get('sequence_id')!r}")

        cam = seq.get("camera") or {}
        if int(cam.get("width_px", -1)) != args.width or int(cam.get("height_px", -1)) != args.height:
            errors.append(f"{sid}: camera resolution {cam.get('width_px')}x{cam.get('height_px')} != {args.width}x{args.height}")
        Kseq = np.asarray(((cam.get("intrinsics") or {}).get("K")), dtype=float)
        if Kseq.shape == (3, 3) and not np.allclose(Kseq, expected_K, atol=1e-4):
            warnings.append(f"{sid}: sequence K differs from nominal AIGP K")

        course_tracks = {g.get("track_id") for g in ((seq.get("course") or {}).get("gate_tracks") or [])}
        last_idx = -1
        last_time = -float("inf")
        for line_no, fr in read_jsonl(frames_jsonl):
            n_frames += 1
            fi = int(fr.get("frame_index", -1))
            if fi <= last_idx:
                errors.append(f"{sid}:{line_no}: frame_index not strictly increasing")
            last_idx = fi
            ts = fr.get("timestamp_ns")
            if ts is not None:
                ts = int(ts)
                if ts <= last_time:
                    errors.append(f"{sid}:{line_no}: timestamp_ns not strictly increasing")
                last_time = ts

            files = fr.get("files") or {}
            rgb_rel = files.get("rgb")
            mask_rel = files.get("instance_mask")
            if not rgb_rel:
                errors.append(f"{sid}:{line_no}: files.rgb missing")
                continue
            rgb_path = seq_dir / rgb_rel
            img = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if img is None:
                errors.append(f"{sid}:{line_no}: missing/unreadable RGB {rgb_rel}")
                continue
            H, W = img.shape[:2]
            if (W, H) != (args.width, args.height):
                errors.append(f"{sid}:{line_no}: RGB is {W}x{H}, expected {args.width}x{args.height}")

            K = np.asarray((((fr.get("camera") or {}).get("intrinsics") or {}).get("K")
                            or ((cam.get("intrinsics") or {}).get("K"))), dtype=float)
            if K.shape != (3, 3):
                errors.append(f"{sid}:{line_no}: camera K missing/invalid")
            elif not np.allclose(K, expected_K, atol=1e-4):
                warnings.append(f"{sid}:{line_no}: frame K differs from nominal AIGP K")

            gates = fr.get("gates") or []
            n_gates += len(gates)
            frame_mask_ids = set()
            for gate in gates:
                tid = gate.get("track_id")
                if tid is not None and course_tracks and tid not in course_tracks:
                    errors.append(f"{sid}:{line_no}: unknown track_id {tid}")
                mid = gate.get("mask_id")
                if mid is not None:
                    mid = int(mid)
                    if mid <= 0 or mid in frame_mask_ids:
                        errors.append(f"{sid}:{line_no}: invalid/duplicate mask_id {mid}")
                    frame_mask_ids.add(mid)
                kp = gate.get("keypoints_2d") or {}
                missing = [k for k in KEYS if k not in kp]
                if missing:
                    errors.append(f"{sid}:{line_no}:{tid}: missing keypoints {missing}")
                pose = gate.get("pose") or {}
                Tcg = pose.get("T_camera_gate")
                if Tcg is not None and not valid_T(Tcg):
                    errors.append(f"{sid}:{line_no}:{tid}: invalid T_camera_gate")

                # Optional exact reprojection check if 3-D camera points exist.
                kp3 = gate.get("keypoints_3d_camera_m") or {}
                if K.shape == (3, 3) and kp3:
                    for key in KEYS:
                        xyz = kp3.get(key)
                        rec2 = kp.get(key) or {}
                        uv = rec2.get("projected_px")
                        if xyz is None or uv is None:
                            continue
                        p = np.asarray(xyz, dtype=float)
                        if p.shape != (3,) or p[2] <= 0:
                            continue
                        q = K @ p
                        q = q[:2] / q[2]
                        if np.linalg.norm(q - np.asarray(uv, dtype=float)) > args.projection_tol_px:
                            errors.append(f"{sid}:{line_no}:{tid}:{key}: reprojection > {args.projection_tol_px}px")

            if mask_rel:
                mask_path = seq_dir / mask_rel
                m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                if m is None:
                    errors.append(f"{sid}:{line_no}: missing/unreadable instance mask {mask_rel}")
                else:
                    n_masks += 1
                    if m.shape[:2] != (H, W):
                        errors.append(f"{sid}:{line_no}: mask dimensions {m.shape[:2]} != RGB {(H,W)}")
                    if m.ndim == 3:
                        m = m[..., 0]
                    ids = set(int(x) for x in np.unique(m) if int(x) != 0)
                    unknown = ids - frame_mask_ids
                    if unknown:
                        errors.append(f"{sid}:{line_no}: mask pixels contain unannotated IDs {sorted(unknown)}")
                    for mid in frame_mask_ids:
                        # A fully occluded gate should normally have mask_id null.
                        if mid not in ids:
                            warnings.append(f"{sid}:{line_no}: annotated mask_id {mid} has zero pixels")

            Twc = (fr.get("camera") or {}).get("T_world_camera")
            Twb = (fr.get("drone") or {}).get("T_world_body")
            if Twc is not None and not valid_T(Twc):
                errors.append(f"{sid}:{line_no}: invalid T_world_camera")
            if Twb is not None and not valid_T(Twb):
                errors.append(f"{sid}:{line_no}: invalid T_world_body")

    print(f"validated sequences={len(sequence_ids)} frames={n_frames} masks={n_masks} gate_records={n_gates}")
    print(f"AIGP intrinsics imply HFoV={hfov:.3f} deg, VFoV={vfov:.3f} deg")
    print(f"nominal K = [[{args.fx},0,{args.cx}],[0,{args.fy},{args.cy}],[0,0,1]]")
    for w in warnings[:50]:
        print(f"WARNING: {w}")
    if len(warnings) > 50:
        print(f"WARNING: ... {len(warnings)-50} additional warnings")
    if errors:
        for e in errors[:100]:
            print(f"ERROR: {e}")
        if len(errors) > 100:
            print(f"ERROR: ... {len(errors)-100} additional errors")
        print(f"Dataset validation FAILED: errors={len(errors)} warnings={len(warnings)}")
        raise SystemExit(1)
    print(f"Dataset validation OK: errors=0 warnings={len(warnings)}")


if __name__ == "__main__":
    main()
