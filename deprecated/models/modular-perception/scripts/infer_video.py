from __future__ import annotations
import argparse
from collections import deque
from pathlib import Path
import json
import time

import cv2
import numpy as np
import torch
from PIL import Image

from gateperception.geometry.camera import camera_matrix, fov_from_intrinsics
from gateperception.geometry.pnp import solve_gate_pose
from gateperception.inference.decoder import decode_segmentation_frame
from gateperception.inference.tracker import GateTracker
from gateperception.inference.keypoints import KeypointTrackMemory
from gateperception.models import build_segmentation_model, build_keypoint_model
from gateperception.utils.config import load_yaml


def load_model(checkpoint: str, builder, device):
    ck = torch.load(checkpoint, map_location="cpu")
    cfg = ck.get("config")
    if cfg is None:
        raise ValueError(f"Checkpoint {checkpoint} has no embedded config")
    model = builder(cfg).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, cfg


def color_for_track(track_id: str | None, mask_id: int | None) -> tuple[int, int, int]:
    # BGR palette; stable across frames when track_id is available.
    palette = [
        (54, 67, 244), (99, 175, 54), (230, 126, 34), (155, 89, 182),
        (26, 188, 156), (241, 196, 15), (52, 152, 219), (231, 76, 60),
        (46, 204, 113), (149, 165, 166), (211, 84, 0), (142, 68, 173),
    ]
    key = track_id or f"mask_{mask_id or 0}"
    idx = sum((i + 1) * ord(c) for i, c in enumerate(key)) % len(palette)
    return palette[idx]


def draw_overlay(frame_bgr: np.ndarray, detections, instances, frame_index: int, total_ms: float, alpha: float) -> np.ndarray:
    out = frame_bgr.copy()
    # Colored mask overlays, one stable color per track.
    for det in detections:
        if det.mask_id is None or not det.mask.any():
            continue
        color = color_for_track(det.track_id, det.mask_id)
        pix = out[det.mask].astype(np.float32)
        col = np.asarray(color, dtype=np.float32)[None, :]
        out[det.mask] = np.clip((1.0 - alpha) * pix + alpha * col, 0, 255).astype(np.uint8)

    for det, inst in zip(detections, instances):
        color = color_for_track(det.track_id, det.mask_id)
        x0, y0, x1, y1 = map(int, map(round, det.box_xyxy_px))
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
        kp = inst.get("keypoints") or {}
        for rec in kp.values():
            xy = rec.get("xy_px")
            if xy is None:
                continue
            x, y = int(round(xy[0])), int(round(xy[1]))
            if 0 <= x < out.shape[1] and 0 <= y < out.shape[0]:
                cv2.circle(out, (x, y), 2, color, -1, cv2.LINE_AA)
        pose = inst.get("pose")
        pose_txt = ""
        if pose is not None:
            pose_txt = f" z={pose.get('depth_camera_z_m', float('nan')):.2f}m d={pose.get('distance_camera_m', float('nan')):.2f}m"
        src = " mem" if det.source == "memory_propagated" else ""
        label = f"{det.track_id or 'track?'}{src} conf={det.score:.2f}{pose_txt}"
        ty = max(11, y0 - 4)
        cv2.putText(out, label, (max(0, x0), ty), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)

    fps_now = 1000.0 / total_ms if total_ms > 1e-6 else 0.0
    num_visible = sum(1 for d in detections if d.mask_id is not None)
    num_memory = sum(1 for d in detections if d.mask_id is None)
    lines = [
        f"frame {frame_index}",
        f"gates {num_visible} + mem {num_memory}",
        f"{total_ms:.1f} ms  {fps_now:.1f} FPS",
    ]
    y = 12
    for text in lines:
        cv2.putText(out, text, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
        y += 12
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--seg-checkpoint", required=True)
    ap.add_argument("--keypoint-checkpoint", required=True)
    ap.add_argument("--camera-config", default="configs/camera.yaml")
    ap.add_argument("--gate-geometry", default="configs/gate_geometry.yaml")
    ap.add_argument("--output", required=True, help="Output directory for JSON/masks/video")
    ap.add_argument("--output-video", default=None, help="Defaults to <output>/overlay.mp4")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--object-threshold", type=float, default=0.5)
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    ap.add_argument("--track-max-age", type=int, default=8)
    ap.add_argument("--resize-input", action="store_true")
    ap.add_argument("--overlay-alpha", type=float, default=0.38)
    args = ap.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    seg, seg_cfg = load_model(args.seg_checkpoint, build_segmentation_model, device)
    kp, kp_cfg = load_model(args.keypoint_checkpoint, build_keypoint_model, device)
    cam_cfg = load_yaml(args.camera_config)["camera"]
    geom_cfg = load_yaml(args.gate_geometry)["gate"]
    W, H = int(cam_cfg["width_px"]), int(cam_cfg["height_px"])
    K = camera_matrix(cam_cfg["fx"], cam_cfg["fy"], cam_cfg["cx"], cam_cfg["cy"])
    distortion = np.asarray(cam_cfg.get("distortion_coefficients", [0,0,0,0,0]), dtype=np.float64)
    hfov, vfov = fov_from_intrinsics(W, H, cam_cfg["fx"], cam_cfg["fy"])

    out = Path(args.output)
    (out / "instance_masks").mkdir(parents=True, exist_ok=True)
    output_video = Path(args.output_video) if args.output_video else out / "overlay.mp4"
    output_video.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(args.video)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0
    src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (src_w, src_h) != (W, H) and not args.resize_input:
        raise ValueError(f"Video is {src_w}x{src_h}, expected calibrated {W}x{H}. Use --resize-input only if intentional.")
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer: {output_video}")

    seg_window = int(seg_cfg["model"].get("window_size", 4))
    rgb_hist: deque[torch.Tensor] = deque(maxlen=seg_window)
    tracker = GateTracker(max_age=args.track_max_age)
    kp_memory = KeypointTrackMemory(kp, device, window_size=int(kp_cfg["model"].get("window_size", 5)), crop_size=tuple(kp_cfg["model"].get("crop_size", [192,192])), crop_padding=float(kp_cfg["model"].get("crop_padding", 0.35)))

    meta = {
        "schema_name": "uav_gate_perception_predictions",
        "schema_version": "1.0.0",
        "model": {"segmentation_checkpoint": args.seg_checkpoint, "keypoint_checkpoint": args.keypoint_checkpoint, "pose_solver": "OpenCV IPPE + LM refinement"},
        "source": {"type": "video", "path": args.video, "source_width_px": src_w, "source_height_px": src_h, "width_px": W, "height_px": H, "fps": fps},
        "camera": {"fx": cam_cfg["fx"], "fy": cam_cfg["fy"], "cx": cam_cfg["cx"], "cy": cam_cfg["cy"], "derived_hfov_deg": hfov, "derived_vfov_deg": vfov},
        "inference": {"device": str(device), "seg_temporal_window_size": seg_window, "keypoint_temporal_window_size": int(kp_cfg["model"].get("window_size", 5)), "object_confidence_threshold": args.object_threshold, "mask_threshold": args.mask_threshold, "tracking_enabled": True},
        "outputs_available": {"instance_segmentation": True, "bounding_boxes": True, "keypoints": True, "keypoint_visibility": True, "pose": True, "tracking": True, "overlay_video": True},
        "overlay_video": str(output_video),
    }
    with open(out / "inference.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    frame_json = open(out / "frames.jsonl", "w", encoding="utf-8")
    frame_index = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            if (bgr.shape[1], bgr.shape[0]) != (W, H):
                bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
            rgb_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb_t = torch.from_numpy(rgb_np.copy()).permute(2,0,1).float() / 255.0
            rgb_hist.append(rgb_t)
            frames = list(rgb_hist)
            while len(frames) < seg_window:
                frames.insert(0, frames[0])
            x = torch.stack(frames, dim=0)[None].to(device)
            t1 = time.perf_counter()
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                raw = seg(x, return_all_frames=False)["frames"][-1]
            raw0 = {k: v[0] for k, v in raw.items()}
            inst, visible_dets = decode_segmentation_frame(raw0, H, W, args.object_threshold, args.mask_threshold)
            visible_dets = tracker.update(frame_index, visible_dets)
            dets = visible_dets + tracker.memory_only(frame_index, max_emit_age=2)
            t2 = time.perf_counter()
            instances = []
            for det in dets:
                kps = kp_memory.predict(rgb_np, det)
                pose = solve_gate_pose(kps, geom_cfg["keypoints_gate_frame_m"], K, distortion)
                instances.append({
                    "mask_id": det.mask_id,
                    "source": det.source,
                    "track_id": det.track_id,
                    "detection_score": det.score,
                    "mask_score": None if det.mask_id is None else det.score,
                    "box_xyxy_px": det.box_xyxy_px,
                    "visible_area_px": int(det.mask.sum()),
                    "keypoints": kps,
                    "pose": pose,
                })
            t3 = time.perf_counter()
            total_ms = (t3 - t0) * 1000.0
            overlay = draw_overlay(bgr, dets, instances, frame_index, total_ms, args.overlay_alpha)
            writer.write(overlay)
            mask_name = f"frame_{frame_index:06d}.png"
            Image.fromarray(inst.astype(np.uint16)).save(out / "instance_masks" / mask_name)
            row = {
                "frame_index": frame_index,
                "timestamp_s": frame_index / fps,
                "instance_mask": f"instance_masks/{mask_name}",
                "num_gate_instances": len(instances),
                "num_mask_instances": sum(1 for x in instances if x["mask_id"] is not None),
                "instances": instances,
                "runtime": {
                    "inference_ms": (t2 - t1) * 1000.0,
                    "postprocess_and_downstream_ms": (t3 - t2) * 1000.0,
                    "total_ms": total_ms,
                },
            }
            frame_json.write(json.dumps(row) + "\n")
            frame_index += 1
    finally:
        frame_json.close()
        cap.release()
        writer.release()

    meta["source"]["num_frames"] = frame_index
    with open(out / "inference.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {frame_index} frames to {out}")
    print(f"Overlay video: {output_video}")


if __name__ == "__main__":
    main()
