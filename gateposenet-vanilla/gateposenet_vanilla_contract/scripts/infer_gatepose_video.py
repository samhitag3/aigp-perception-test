#!/usr/bin/env python3
"""Run unchanged GatePoseNetSingle on a video and write canonical predictions.

Because the original architecture is single-target and has a one-channel
auxiliary segmentation head, inference honestly declares segmentation_type
"union" and emits one target_prediction per frame (not fake instances).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatenet.utils import load_checkpoint, load_config, pick_device
from gateposenet.model_single import build_gateposenet_single

OUTER_KEYS = ("outer_tl", "outer_tr", "outer_br", "outer_bl")


def quat_xyzw_from_R(R: np.ndarray) -> list[float]:
    # Numerically stable enough for serialization/debugging; T_camera_gate is
    # the authoritative pose representation.
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
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif i == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= max(np.linalg.norm(q), 1e-12)
    return [float(x) for x in q]


def read_ego(path: str | None) -> dict[int, np.ndarray]:
    if not path:
        return {}
    out = {}
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
            if a.shape == (6,):
                out[idx] = a
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/gateposenet_vanilla_contract.yaml")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--presence-th", type=float, default=0.5)
    ap.add_argument("--mask-th", type=float, default=0.5)
    ap.add_argument("--ego-jsonl", default=None,
                    help="optional frame_index + ego_motion_cam[6] JSONL; otherwise v/omega=0")
    ap.add_argument("--save-overlays", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = cfg["data"]
    cam_cfg = cfg.get("camera", {})
    device = pick_device(args.device)
    model = build_gateposenet_single(cfg["model"]).to(device).eval()
    load_checkpoint(args.checkpoint, model, map_location=device)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    expected_w = int(cam_cfg.get("width_px", d.get("native_width", 640)))
    expected_h = int(cam_cfg.get("height_px", d.get("native_height", 360)))
    if (src_w, src_h) != (expected_w, expected_h):
        print(f"WARNING: video is {src_w}x{src_h}; training camera is {expected_w}x{expected_h}. "
              "Direct metric pose assumes the same camera/FoV unless you intentionally calibrated for this input.")

    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs") / Path(args.video).stem
    masks_dir = out_dir / "union_masks"
    overlays_dir = out_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        overlays_dir.mkdir(parents=True, exist_ok=True)

    ego_map = read_ego(args.ego_jsonl)
    h_model, w_model = int(d.get("height", 192)), int(d.get("width", 320))
    dt = 1.0 / max(fps, 1e-6)
    hidden = None
    frame_records = []
    timings = []

    idx = 0
    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames is not None and idx >= args.max_frames:
                break

            rgb = cv2.cvtColor(cv2.resize(frame, (w_model, h_model), interpolation=cv2.INTER_AREA),
                               cv2.COLOR_BGR2RGB)
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

            t1 = time.perf_counter()
            vis_score = float(torch.sigmoid(pred["visible_logit"])[0].item())
            visible_frac = float(pred["visible_frac"][0].item())
            corners_uv = pred["corners_uv"][0].detach().cpu().numpy()
            corners_px = corners_uv * np.array([src_w, src_h], dtype=np.float32)
            corner_inside_score = torch.sigmoid(pred["corner_inside_logit"])[0].detach().cpu().numpy()
            center_uv = pred["center_uv"][0].detach().cpu().numpy()
            center_px = center_uv * np.array([src_w, src_h], dtype=np.float32)
            position = pred["position"][0].detach().cpu().numpy().astype(np.float64)
            R = pred["R"][0].detach().cpu().numpy().astype(np.float64)
            Tcg = np.eye(4, dtype=np.float64)
            Tcg[:3, :3] = R
            Tcg[:3, 3] = position

            seg_prob = torch.sigmoid(pred["seg_logit"])
            seg_full = F.interpolate(seg_prob, size=(src_h, src_w), mode="bilinear", align_corners=False)[0, 0]
            union = (seg_full >= args.mask_th).to(torch.uint8).cpu().numpy()
            mask_rel = f"union_masks/frame_{idx:06d}.png"
            cv2.imwrite(str(out_dir / mask_rel), union)

            keypoints = {}
            for k, xy, score in zip(OUTER_KEYS, corners_px, corner_inside_score):
                keypoints[k] = {
                    "xy_px": [float(xy[0]), float(xy[1])],
                    "confidence": float(score),
                    "inside_frame_probability": float(score),
                }

            post_ms = (time.perf_counter() - t1) * 1000.0
            total_ms = infer_ms + post_ms
            timings.append(total_ms)
            record = {
                "frame_index": idx,
                "timestamp_s": float(idx / fps),
                "segmentation_type": "union",
                "union_mask": mask_rel,
                "instances": None,
                "target_prediction": {
                    "track_id": None,
                    "detection_score": vis_score,
                    "predicted_visible": bool(vis_score >= args.presence_th),
                    "visible_fraction": visible_frac,
                    "center_xy_px": [float(center_px[0]), float(center_px[1])],
                    "keypoints": keypoints,
                    "inner_keypoints": None,
                    "pose": {
                        "T_camera_gate": Tcg.tolist(),
                        "translation_camera_m": [float(x) for x in position],
                        "quaternion_camera_xyzw": quat_xyzw_from_R(R),
                        "distance_camera_m": float(np.linalg.norm(position)),
                        "depth_camera_z_m": float(position[2]),
                        "confidence": vis_score,
                    },
                },
                "runtime": {
                    "inference_ms": infer_ms,
                    "postprocess_ms": post_ms,
                    "total_ms": total_ms,
                },
            }
            frame_records.append(record)

            if args.save_overlays:
                overlay = frame.copy()
                mask_bool = union.astype(bool)
                tint = overlay.copy()
                tint[mask_bool] = (0, 180, 255)
                overlay = cv2.addWeighted(overlay, 0.7, tint, 0.3, 0)
                for xy in corners_px:
                    cv2.circle(overlay, (int(round(xy[0])), int(round(xy[1]))), 4, (0, 0, 255), -1)
                cv2.circle(overlay, (int(round(center_px[0])), int(round(center_px[1]))), 5, (255, 0, 0), -1)
                cv2.putText(overlay, f"target vis={vis_score:.2f} z={position[2]:.2f}m",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imwrite(str(overlays_dir / f"frame_{idx:06d}.jpg"), overlay)
            idx += 1

    cap.release()

    inference_meta = {
        "schema_name": "uav_gate_perception_predictions",
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "model_name": "GatePoseNetSingle",
            "model_family": "original_gateposenet_single_target",
            "architecture_modified": False,
            "checkpoint": str(args.checkpoint),
        },
        "source": {
            "type": "video",
            "path": str(args.video),
            "width_px": src_w,
            "height_px": src_h,
            "fps": fps,
            "num_frames": idx,
            "reported_num_frames": num_frames_meta,
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
            "ego_motion_source": "ego_jsonl" if args.ego_jsonl else "zero_v_omega_plus_video_dt",
            "presence_threshold": args.presence_th,
            "mask_threshold": args.mask_th,
            "mean_total_ms": float(np.mean(timings)) if timings else None,
            "median_total_ms": float(np.median(timings)) if timings else None,
            "mean_fps": float(1000.0 / np.mean(timings)) if timings and np.mean(timings) > 0 else None,
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
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "inference.json").open("w", encoding="utf-8") as f:
        json.dump(inference_meta, f, indent=2, allow_nan=False)
        f.write("\n")
    with (out_dir / "frames.jsonl").open("w", encoding="utf-8") as f:
        for rec in frame_records:
            f.write(json.dumps(rec, separators=(",", ":"), allow_nan=False) + "\n")

    print(f"frames={idx} output={out_dir}")
    print(f"wrote {out_dir / 'inference.json'}")
    print(f"wrote {out_dir / 'frames.jsonl'}")
    print(f"union masks -> {masks_dir}")


if __name__ == "__main__":
    main()
