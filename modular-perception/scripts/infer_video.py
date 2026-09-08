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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--seg-checkpoint", required=True)
    ap.add_argument("--keypoint-checkpoint", required=True)
    ap.add_argument("--camera-config", default="configs/camera.yaml")
    ap.add_argument("--gate-geometry", default="configs/gate_geometry.yaml")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--object-threshold", type=float, default=0.5)
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    ap.add_argument("--track-max-age", type=int, default=8)
    ap.add_argument("--resize-input", action="store_true")
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

    out = Path(args.output); (out / "instance_masks").mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(args.video)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (src_w, src_h) != (W, H) and not args.resize_input:
        raise ValueError(f"Video is {src_w}x{src_h}, expected calibrated {W}x{H}. Use --resize-input only if intentional.")

    seg_window = int(seg_cfg["model"].get("window_size", 4))
    rgb_hist: deque[torch.Tensor] = deque(maxlen=seg_window)
    tracker = GateTracker(max_age=args.track_max_age)
    kp_memory = KeypointTrackMemory(kp, device, window_size=int(kp_cfg["model"].get("window_size", 5)), crop_size=tuple(kp_cfg["model"].get("crop_size", [192,192])), crop_padding=float(kp_cfg["model"].get("crop_padding", 0.35)))

    meta = {
        "schema_name": "uav_gate_perception_predictions",
        "schema_version": "1.0.0",
        "model": {"segmentation_checkpoint": args.seg_checkpoint, "keypoint_checkpoint": args.keypoint_checkpoint, "pose_solver": "OpenCV IPPE + LM refinement"},
        "source": {"type": "video", "path": args.video, "width_px": W, "height_px": H, "fps": fps},
        "camera": {"fx": cam_cfg["fx"], "fy": cam_cfg["fy"], "cx": cam_cfg["cx"], "cy": cam_cfg["cy"], "derived_hfov_deg": hfov, "derived_vfov_deg": vfov},
        "inference": {"device": str(device), "seg_temporal_window_size": seg_window, "keypoint_temporal_window_size": int(kp_cfg["model"].get("window_size", 5)), "object_confidence_threshold": args.object_threshold, "mask_threshold": args.mask_threshold, "tracking_enabled": True},
        "outputs_available": {"instance_segmentation": True, "bounding_boxes": True, "keypoints": True, "keypoint_visibility": True, "pose": True, "tracking": True}
    }
    with open(out / "inference.json", "w", encoding="utf-8") as f: json.dump(meta, f, indent=2)

    frame_json = open(out / "frames.jsonl", "w", encoding="utf-8")
    frame_index = 0
    while True:
        ok, bgr = cap.read()
        if not ok: break
        t0 = time.perf_counter()
        if (bgr.shape[1], bgr.shape[0]) != (W, H):
            bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
        rgb_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_t = torch.from_numpy(rgb_np.copy()).permute(2,0,1).float() / 255.0
        rgb_hist.append(rgb_t)
        frames = list(rgb_hist)
        while len(frames) < seg_window: frames.insert(0, frames[0])
        x = torch.stack(frames, dim=0)[None].to(device)
        t1 = time.perf_counter()
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            raw = seg(x, return_all_frames=False)["frames"][-1]
        raw0 = {k: v[0] for k, v in raw.items()}
        inst, dets = decode_segmentation_frame(raw0, H, W, args.object_threshold, args.mask_threshold)
        dets = tracker.update(frame_index, dets)
        dets = dets + tracker.memory_only(frame_index, max_emit_age=2)
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
        mask_name = f"frame_{frame_index:06d}.png"
        Image.fromarray(inst, mode="I;16").save(out / "instance_masks" / mask_name)
        row = {
            "frame_index": frame_index,
            "timestamp_s": frame_index / fps if fps > 0 else None,
            "instance_mask": f"instance_masks/{mask_name}",
            "num_gate_instances": len(instances),
            "num_mask_instances": sum(1 for x in instances if x["mask_id"] is not None),
            "instances": instances,
            "runtime": {
                "inference_ms": (t2 - t1) * 1000.0,
                "postprocess_and_downstream_ms": (t3 - t2) * 1000.0,
                "total_ms": (t3 - t0) * 1000.0,
            }
        }
        frame_json.write(json.dumps(row) + "\n")
        frame_index += 1
    frame_json.close(); cap.release()
    meta["source"]["num_frames"] = frame_index
    with open(out / "inference.json", "w", encoding="utf-8") as f: json.dump(meta, f, indent=2)
    print(f"Wrote {frame_index} frames to {out}")

if __name__ == "__main__":
    main()
