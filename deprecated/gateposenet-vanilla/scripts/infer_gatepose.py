#!/usr/bin/env python3
"""Canonical visual inference for unchanged vanilla GatePoseNetSingle.

Supports either:
  * a video file, or
  * a lexicographically/naturally sorted folder of images.

Outputs:
  inference.json
  frames.jsonl
  union_masks/frame_XXXXXX.png
  overlays/frame_XXXXXX.jpg
  overlay.mp4

The legacy GatePoseNetSingle remains single-target and union-segmentation only.
This script does not invent per-gate instance masks or inner-corner predictions.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatenet.utils import load_checkpoint, load_config, pick_device
from gateposenet.model_single import build_gateposenet_single

OUTER_KEYS = ("outer_tl", "outer_tr", "outer_br", "outer_bl")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def natural_key(path: Path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]


def quat_xyzw_from_R(R: np.ndarray) -> list[float]:
    m = R.astype(np.float64)
    tr = float(np.trace(m))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = math.sqrt(max(1e-12, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) * 2
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif i == 1:
            s = math.sqrt(max(1e-12, 1.0 + m[1, 1] - m[0, 0] - m[2, 2])) * 2
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(max(1e-12, 1.0 + m[2, 2] - m[0, 0] - m[1, 1])) * 2
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= max(np.linalg.norm(q), 1e-12)
    return [float(x) for x in q]


def rpy_deg_from_R(R: np.ndarray) -> tuple[float, float, float]:
    """XYZ fixed-axis roll/pitch/yaw summary for visualization only."""
    r20 = float(R[2, 0])
    pitch = math.asin(max(-1.0, min(1.0, -r20)))
    cp = math.cos(pitch)
    if abs(cp) > 1e-6:
        roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
        yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:
        roll = math.atan2(-float(R[1, 2]), float(R[1, 1]))
        yaw = 0.0
    return tuple(math.degrees(v) for v in (roll, pitch, yaw))


def read_ego(path: str | None) -> dict[int, np.ndarray]:
    if not path:
        return {}
    out: dict[int, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            idx = int(r["frame_index"])
            x = r.get("ego_motion_cam") or r.get("ego")
            if x is None:
                continue
            a = np.asarray(x, dtype=np.float32)
            if a.shape != (6,):
                raise ValueError(f"ego vector for frame {idx} must have shape (6,), got {a.shape}")
            out[idx] = a
    return out


def iter_images(folder: Path) -> tuple[list[Path], tuple[int, int]]:
    paths = sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES],
        key=natural_key,
    )
    if not paths:
        raise FileNotFoundError(f"no supported images found in {folder}")
    first = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"failed to read first image: {paths[0]}")
    h, w = first.shape[:2]
    return paths, (w, h)


def frame_source(input_path: Path, image_fps: float):
    """Return iterator factory metadata and a cleanup callback."""
    if input_path.is_dir():
        paths, (w, h) = iter_images(input_path)

        def iterator() -> Iterator[tuple[int, np.ndarray, str]]:
            for i, p in enumerate(paths):
                frame = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError(f"failed to read image: {p}")
                yield i, frame, p.name

        return {
            "kind": "images",
            "iterator": iterator,
            "cleanup": lambda: None,
            "fps": float(image_fps),
            "width": w,
            "height": h,
            "reported_num_frames": len(paths),
        }

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {input_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or image_fps)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def iterator() -> Iterator[tuple[int, np.ndarray, str]]:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield i, frame, f"frame_{i:06d}"
            i += 1

    return {
        "kind": "video",
        "iterator": iterator,
        "cleanup": cap.release,
        "fps": fps,
        "width": w,
        "height": h,
        "reported_num_frames": n,
    }


def translucent_gate_overlay(frame: np.ndarray, union: np.ndarray, detected: bool,
                             alpha: float) -> np.ndarray:
    """Tint only predicted gate pixels while preserving the original background."""
    out = frame.copy()
    mask = union.astype(bool)
    if not np.any(mask):
        return out
    # Green = accepted target; orange = below presence threshold / uncertain.
    color = np.array((40, 220, 80) if detected else (0, 165, 255), dtype=np.float32)  # BGR
    pix = out[mask].astype(np.float32)
    out[mask] = np.clip((1.0 - alpha) * pix + alpha * color, 0, 255).astype(np.uint8)
    return out


def put_hud(frame: np.ndarray, lines: list[str]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thickness = 1
    line_h = 22
    pad = 10
    widths = [cv2.getTextSize(x, font, scale, thickness)[0][0] for x in lines]
    box_w = min(frame.shape[1] - 12, max(widths, default=0) + 2 * pad)
    box_h = len(lines) * line_h + 2 * pad
    x0, y0 = 6, 6
    x1, y1 = x0 + box_w, min(frame.shape[0] - 6, y0 + box_h)
    roi = frame[y0:y1, x0:x1]
    if roi.size:
        dark = np.zeros_like(roi)
        cv2.addWeighted(roi, 0.35, dark, 0.65, 0, dst=roi)
    y = y0 + pad + 15
    for i, text in enumerate(lines):
        # Title/status gets heavier white; metrics use light text.
        thick = 2 if i == 0 else 1
        cv2.putText(frame, text, (x0 + pad, y), font, scale, (245, 245, 245), thick,
                    cv2.LINE_AA)
        y += line_h


def draw_geometry(frame: np.ndarray, corners_px: np.ndarray, corner_scores: np.ndarray,
                  center_px: np.ndarray, detected: bool) -> None:
    pts = np.round(corners_px).astype(np.int32)
    if np.all(np.isfinite(corners_px)):
        cv2.polylines(frame, [pts.reshape(-1, 1, 2)], True,
                      (255, 255, 0) if detected else (0, 190, 255), 2, cv2.LINE_AA)
    for xy, score in zip(pts, corner_scores):
        color = (0, 255, 255) if score >= 0.5 else (0, 90, 255)
        cv2.circle(frame, (int(xy[0]), int(xy[1])), 4, color, -1, cv2.LINE_AA)
    c = np.round(center_px).astype(np.int32)
    cv2.drawMarker(frame, (int(c[0]), int(c[1])), (255, 80, 255),
                   markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True,
                    help="video file OR folder of sequential images")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--label", default=None,
                    help="short stage label shown in the overlay HUD")
    ap.add_argument("--presence-th", type=float, default=0.5)
    ap.add_argument("--mask-th", type=float, default=0.5)
    ap.add_argument("--overlay-alpha", type=float, default=0.38)
    ap.add_argument("--image-fps", type=float, default=30.0,
                    help="timeline/output-video FPS when --input is an image folder")
    ap.add_argument("--ego-jsonl", default=None,
                    help="optional frame_index + ego_motion_cam[6] JSONL; otherwise zero v/omega")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--show", action="store_true",
                    help="live cv2 window; avoid on a headless remote server")
    ap.add_argument("--no-overlay-video", action="store_true")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    if not (0.0 <= args.overlay_alpha <= 1.0):
        raise ValueError("--overlay-alpha must be in [0,1]")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    cfg = load_config(args.config)
    d = cfg["data"]
    cam_cfg = cfg.get("camera", {})
    device = pick_device(args.device)
    model = build_gateposenet_single(cfg["model"]).to(device).eval()
    load_checkpoint(args.checkpoint, model, map_location=device)

    source = frame_source(input_path, args.image_fps)
    fps = float(source["fps"])
    src_w, src_h = int(source["width"]), int(source["height"])
    expected_w = int(cam_cfg.get("width_px", d.get("native_width", 640)))
    expected_h = int(cam_cfg.get("height_px", d.get("native_height", 360)))
    if (src_w, src_h) != (expected_w, expected_h):
        print(
            f"WARNING: input is {src_w}x{src_h}; training camera is {expected_w}x{expected_h}. "
            "Metric pose assumes matching intrinsics/FoV unless this input was calibrated accordingly."
        )

    out_dir = Path(args.out_dir)
    masks_dir = out_dir / "union_masks"
    overlays_dir = out_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    video_writer = None
    overlay_video_path = out_dir / "overlay.mp4"
    if not args.no_overlay_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(overlay_video_path), fourcc, fps, (src_w, src_h))
        if not video_writer.isOpened():
            raise RuntimeError(f"failed to open output video writer: {overlay_video_path}")

    ego_map = read_ego(args.ego_jsonl)
    h_model, w_model = int(d.get("height", 192)), int(d.get("width", 320))
    dt = 1.0 / max(fps, 1e-6)
    hidden = None
    frame_records: list[dict] = []
    total_timings: list[float] = []
    infer_timings: list[float] = []

    try:
        with torch.no_grad():
            for idx, frame, source_name in source["iterator"]():
                if args.max_frames is not None and idx >= args.max_frames:
                    break
                if frame.shape[1] != src_w or frame.shape[0] != src_h:
                    raise ValueError(
                        f"frame {source_name} is {frame.shape[1]}x{frame.shape[0]}, expected {src_w}x{src_h}; "
                        "all frames in one inference sequence must share dimensions"
                    )

                rgb = cv2.cvtColor(
                    cv2.resize(frame, (w_model, h_model), interpolation=cv2.INTER_AREA),
                    cv2.COLOR_BGR2RGB,
                )
                image = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
                ego6 = ego_map.get(idx, np.zeros(6, dtype=np.float32))
                ego7 = np.concatenate([ego6, [np.float32(dt)]])
                ego = torch.from_numpy(ego7).unsqueeze(0).to(device)

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                pred, hidden = model.step(image, ego, hidden)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                infer_ms = (time.perf_counter() - t0) * 1000.0

                p0 = time.perf_counter()
                vis_score = float(torch.sigmoid(pred["visible_logit"])[0].item())
                detected = vis_score >= args.presence_th
                visible_frac = float(pred["visible_frac"][0].item())
                corners_uv = pred["corners_uv"][0].detach().cpu().numpy()
                corners_px = corners_uv * np.array([src_w, src_h], dtype=np.float32)
                corner_scores = torch.sigmoid(pred["corner_inside_logit"])[0].detach().cpu().numpy()
                center_uv = pred["center_uv"][0].detach().cpu().numpy()
                center_px = center_uv * np.array([src_w, src_h], dtype=np.float32)
                position = pred["position"][0].detach().cpu().numpy().astype(np.float64)
                R = pred["R"][0].detach().cpu().numpy().astype(np.float64)
                rpy = rpy_deg_from_R(R)
                Tcg = np.eye(4, dtype=np.float64)
                Tcg[:3, :3] = R
                Tcg[:3, 3] = position

                seg_prob = torch.sigmoid(pred["seg_logit"])
                seg_full = F.interpolate(
                    seg_prob, size=(src_h, src_w), mode="bilinear", align_corners=False
                )[0, 0]
                union = (seg_full >= args.mask_th).to(torch.uint8).cpu().numpy()
                mask_area_px = int(union.sum())
                mask_area_frac = float(mask_area_px / max(1, src_w * src_h))
                mask_rel = f"union_masks/frame_{idx:06d}.png"
                cv2.imwrite(str(out_dir / mask_rel), union)

                keypoints = {}
                for k, xy, score in zip(OUTER_KEYS, corners_px, corner_scores):
                    keypoints[k] = {
                        "xy_px": [float(xy[0]), float(xy[1])],
                        "confidence": float(score),
                        "inside_frame_probability": float(score),
                    }

                post_ms = (time.perf_counter() - p0) * 1000.0
                total_ms = infer_ms + post_ms
                infer_timings.append(infer_ms)
                total_timings.append(total_ms)
                inst_fps = 1000.0 / total_ms if total_ms > 0 else float("inf")
                distance = float(np.linalg.norm(position))

                overlay = translucent_gate_overlay(frame, union, detected, args.overlay_alpha)
                draw_geometry(overlay, corners_px, corner_scores, center_px, detected)
                stage = args.label or Path(args.checkpoint).parent.name
                status = "DETECTED" if detected else "LOW CONF / NO TARGET"
                lines = [
                    f"{stage} | {status}",
                    f"frame {idx:06d}  conf {vis_score:.3f}  visible-frac {visible_frac:.3f}",
                    f"mask {100.0 * mask_area_frac:.1f}%  center ({center_px[0]:.1f}, {center_px[1]:.1f}) px",
                    f"camera xyz [{position[0]:+.2f}, {position[1]:+.2f}, {position[2]:+.2f}] m  range {distance:.2f} m",
                    f"r/p/y [{rpy[0]:+.1f}, {rpy[1]:+.1f}, {rpy[2]:+.1f}] deg",
                    f"latency {total_ms:.2f} ms  model {infer_ms:.2f} ms  throughput {inst_fps:.1f} FPS",
                ]
                put_hud(overlay, lines)

                overlay_rel = f"overlays/frame_{idx:06d}.jpg"
                cv2.imwrite(str(out_dir / overlay_rel), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                if video_writer is not None:
                    video_writer.write(overlay)
                if args.show:
                    cv2.imshow("GatePoseNet vanilla inference", overlay)
                    if (cv2.waitKey(max(1, int(round(1000.0 / fps)))) & 0xFF) == 27:
                        break

                frame_records.append({
                    "frame_index": idx,
                    "source_name": source_name,
                    "timestamp_s": float(idx / fps),
                    "segmentation_type": "union",
                    "union_mask": mask_rel,
                    "overlay": overlay_rel,
                    "instances": None,
                    "target_prediction": {
                        "track_id": None,
                        "detection_score": vis_score,
                        "predicted_visible": bool(detected),
                        "visible_fraction": visible_frac,
                        "mask_area_px": mask_area_px,
                        "mask_area_fraction": mask_area_frac,
                        "center_xy_px": [float(center_px[0]), float(center_px[1])],
                        "keypoints": keypoints,
                        "inner_keypoints": None,
                        "pose": {
                            "T_camera_gate": Tcg.tolist(),
                            "translation_camera_m": [float(x) for x in position],
                            "quaternion_camera_xyzw": quat_xyzw_from_R(R),
                            "rpy_camera_deg_visualization": [float(x) for x in rpy],
                            "distance_camera_m": distance,
                            "depth_camera_z_m": float(position[2]),
                            "confidence": vis_score,
                        },
                    },
                    "runtime": {
                        "inference_ms": infer_ms,
                        "postprocess_ms": post_ms,
                        "total_ms": total_ms,
                        "instantaneous_fps": inst_fps,
                    },
                })
    finally:
        source["cleanup"]()
        if video_writer is not None:
            video_writer.release()
        if args.show:
            cv2.destroyAllWindows()

    mean_total = float(np.mean(total_timings)) if total_timings else None
    inference_meta = {
        "schema_name": "uav_gate_perception_predictions",
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "model_name": "GatePoseNetSingle",
            "model_family": "original_gateposenet_single_target",
            "architecture_modified": False,
            "checkpoint": str(args.checkpoint),
            "config": str(args.config),
            "label": args.label,
        },
        "source": {
            "type": source["kind"],
            "path": str(input_path),
            "width_px": src_w,
            "height_px": src_h,
            "fps": fps,
            "num_frames": len(frame_records),
            "reported_num_frames": int(source["reported_num_frames"]),
        },
        "camera": {
            "training_width_px": expected_w,
            "training_height_px": expected_h,
            "fx": float(cam_cfg.get("fx", 320.0)),
            "fy": float(cam_cfg.get("fy", 320.0)),
            "cx": float(cam_cfg.get("cx", 320.0)),
            "cy": float(cam_cfg.get("cy", 180.0)),
            "hfov_deg": float(cam_cfg.get("hfov_deg", 90.0)),
            "vfov_deg": float(cam_cfg.get("vfov_deg", 58.71550708558255)),
        },
        "inference": {
            "device": str(device),
            "temporal": True,
            "ego_motion_source": "ego_jsonl" if args.ego_jsonl else "zero_v_omega_plus_frame_dt",
            "presence_threshold": args.presence_th,
            "mask_threshold": args.mask_th,
            "overlay_alpha": args.overlay_alpha,
            "mean_inference_ms": float(np.mean(infer_timings)) if infer_timings else None,
            "median_inference_ms": float(np.median(infer_timings)) if infer_timings else None,
            "mean_total_ms": mean_total,
            "median_total_ms": float(np.median(total_timings)) if total_timings else None,
            "mean_pipeline_fps": float(1000.0 / mean_total) if mean_total and mean_total > 0 else None,
        },
        "visualization": {
            "overlay_video": None if args.no_overlay_video else "overlay.mp4",
            "overlay_frames": "overlays/",
            "hud_fields": [
                "stage/status", "detection confidence", "visible fraction", "mask area",
                "predicted center", "camera xyz", "range", "roll/pitch/yaw", "latency/FPS"
            ],
        },
        "outputs_available": {
            "union_segmentation": True,
            "instance_segmentation": False,
            "outer_keypoints": True,
            "inner_keypoints": False,
            "pose": True,
            "tracking": False,
        },
        "limitations": [
            "vanilla GatePoseNetSingle predicts one current target, not all gates",
            "segmentation head is union-only",
            "only four outer corners are predicted",
            "HUD values are predictions/runtime statistics, not accuracy metrics unless GT is separately evaluated",
        ],
    }

    with (out_dir / "inference.json").open("w", encoding="utf-8") as f:
        json.dump(inference_meta, f, indent=2, allow_nan=False)
        f.write("\n")
    with (out_dir / "frames.jsonl").open("w", encoding="utf-8") as f:
        for rec in frame_records:
            f.write(json.dumps(rec, separators=(",", ":"), allow_nan=False) + "\n")

    print(f"frames={len(frame_records)} output={out_dir}")
    print(f"wrote {out_dir / 'inference.json'}")
    print(f"wrote {out_dir / 'frames.jsonl'}")
    print(f"union masks -> {masks_dir}")
    print(f"overlay frames -> {overlays_dir}")
    if not args.no_overlay_video:
        print(f"overlay video -> {overlay_video_path}")


if __name__ == "__main__":
    main()
