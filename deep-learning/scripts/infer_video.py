#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from gateperceiver.models import GatePerceiver
from gateperceiver.inference import (
    decode_frame,
    OnlineTracker,
    PredictionWriter,
    draw_prediction_overlay,
    resize_decoded_prediction,
)
from gateperceiver.config import load_yaml
from gateperceiver.contracts import SCHEMA_VERSION


CANONICAL_WIDTH = 640
CANONICAL_HEIGHT = 360
CANONICAL_FX = 320.0
CANONICAL_FY = 320.0
CANONICAL_CX = 320.0
CANONICAL_CY = 180.0


def canonical_intrinsics_for_size(width: int, height: int) -> np.ndarray:
    """Scale the real-camera calibration to a resized image."""
    sx = float(width) / CANONICAL_WIDTH
    sy = float(height) / CANONICAL_HEIGHT
    return np.array(
        [
            [CANONICAL_FX * sx, 0.0, CANONICAL_CX * sx],
            [0.0, CANONICAL_FY * sy, CANONICAL_CY * sy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def main():
    ap = argparse.ArgumentParser(description="GatePerceiver video -> canonical predictions + annotated MP4")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--config", help="Optional config override; checkpoint config is used by default")
    ap.add_argument("--no-tracking", action="store_true")
    ap.add_argument("--overlay-video", default="overlay.mp4", help="Annotated video filename inside --output")
    ap.add_argument("--overlay-alpha", type=float, default=0.36)
    ap.add_argument("--no-keypoints", action="store_true", help="Do not draw predicted keypoints on overlay")
    args = ap.parse_args()

    dev = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location="cpu")
    cfg = load_yaml(args.config) if args.config else ck["config"]
    model = GatePerceiver(cfg)
    model.load_state_dict(ck["model"])
    model.to(dev).eval()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_h, out_w = map(int, cfg["data"].get("image_size", [CANONICAL_HEIGHT, CANONICAL_WIDTH]))
    temporal_window = int(cfg["data"].get("window_size", 1))
    K_model = canonical_intrinsics_for_size(out_w, out_h)

    tracker = None if args.no_tracking else OnlineTracker()
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    overlay_path = out_root / args.overlay_video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(overlay_path), fourcc, source_fps, (src_w, src_h))
    if not video_writer.isOpened():
        raise RuntimeError(f"Could not create output video: {overlay_path}")

    metadata = {
        "schema_name": "uav_gate_perception_predictions",
        "schema_version": SCHEMA_VERSION,
        "model": {"checkpoint": str(args.checkpoint), "architecture": "GatePerceiver"},
        "source": {
            "type": "video",
            "path": str(args.video),
            "width_px": src_w,
            "height_px": src_h,
            "fps": source_fps,
            "num_frames": num_frames,
        },
        "camera": {
            "authoritative_intrinsics": {
                "resolution_px": [CANONICAL_WIDTH, CANONICAL_HEIGHT],
                "fx_fy": [CANONICAL_FX, CANONICAL_FY],
                "cx_cy": [CANONICAL_CX, CANONICAL_CY],
                "hfov_deg": 90.0,
                "vfov_deg": 58.7155,
            },
            "model_input_K": K_model.tolist(),
        },
        "inference": {
            "device": str(dev),
            "model_input_size_px": [out_w, out_h],
            "temporal_window_size": temporal_window,
            "tracking_enabled": tracker is not None,
            "overlay_alpha": float(args.overlay_alpha),
        },
        "outputs_available": {
            "instance_segmentation": True,
            "bounding_boxes": True,
            "keypoints": True,
            "keypoint_visibility": True,
            "pose": True,
            "tracking": tracker is not None,
            "annotated_video": True,
        },
        "artifacts": {"annotated_video": args.overlay_video},
    }

    prediction_writer = PredictionWriter(out_root, metadata)
    window: list[torch.Tensor] = []
    frame_index = 0

    try:
        with torch.no_grad():
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                resized_bgr = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                tensor = torch.from_numpy(rgb).permute(2, 0, 1)
                window.append(tensor)
                window = window[-temporal_window:]
                while len(window) < temporal_window:
                    window.insert(0, window[0])

                inp = torch.stack(window)[None].to(dev)
                K_in = torch.from_numpy(np.stack([K_model] * temporal_window))[None].to(dev)

                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                raw = model(inp, K_in)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                inference_ms = (time.perf_counter() - t0) * 1000.0

                t1 = time.perf_counter()
                mask, instances = decode_frame(raw, 0, temporal_window - 1, cfg)
                if tracker is not None:
                    instances = tracker.update(instances)
                else:
                    for ins in instances:
                        ins["track_id"] = None
                        ins.pop("track_embedding", None)

                # Canonical persisted predictions are expressed in source-video pixel coordinates.
                mask, instances = resize_decoded_prediction(mask, instances, src_w, src_h)
                postprocess_ms = (time.perf_counter() - t1) * 1000.0
                runtime = {
                    "inference_ms": inference_ms,
                    "postprocess_ms": postprocess_ms,
                    "total_ms": inference_ms + postprocess_ms,
                }

                prediction_writer.write_frame(
                    frame_index,
                    frame_index / source_fps,
                    mask,
                    instances,
                    runtime,
                )

                annotated = draw_prediction_overlay(
                    frame,
                    mask,
                    instances,
                    runtime,
                    frame_index,
                    source_fps,
                    alpha=args.overlay_alpha,
                    draw_keypoints=not args.no_keypoints,
                )
                video_writer.write(annotated)
                frame_index += 1
    finally:
        prediction_writer.close()
        cap.release()
        video_writer.release()

    print(f"wrote {frame_index} frames")
    print(f"canonical predictions: {out_root}")
    print(f"annotated video: {overlay_path}")


if __name__ == "__main__":
    main()
