from __future__ import annotations

import hashlib
import math
from typing import Iterable

import cv2
import numpy as np


# BGR palette chosen to remain visually distinct on typical RGB video.
_PALETTE = [
    (60, 80, 255),    # red/orange
    (70, 210, 70),    # green
    (255, 110, 60),   # blue
    (70, 220, 220),   # yellow
    (220, 80, 220),   # magenta
    (220, 180, 60),   # cyan-ish
    (120, 80, 255),   # pink/red
    (80, 170, 255),   # orange
    (200, 120, 80),   # blue-gray
    (120, 220, 160),  # mint
    (230, 140, 230),  # light magenta
    (180, 220, 80),   # lime/cyan
]


def _stable_index(value: str | int | None) -> int:
    if value is None:
        return 0
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little")


def instance_color(instance: dict) -> tuple[int, int, int]:
    """Return a deterministic BGR color, preferring persistent track identity."""
    key = instance.get("track_id") or instance.get("mask_id")
    return _PALETTE[_stable_index(key) % len(_PALETTE)]


def resize_decoded_prediction(
    mask: np.ndarray,
    instances: list[dict],
    dst_width: int,
    dst_height: int,
) -> tuple[np.ndarray, list[dict]]:
    """Map decoded mask/box/keypoint coordinates to a target image resolution."""
    src_h, src_w = mask.shape[:2]
    if src_w == dst_width and src_h == dst_height:
        return mask.astype(np.uint16, copy=False), instances

    sx = float(dst_width) / float(src_w)
    sy = float(dst_height) / float(src_h)
    resized = cv2.resize(mask.astype(np.uint16), (dst_width, dst_height), interpolation=cv2.INTER_NEAREST)

    scaled: list[dict] = []
    for ins in instances:
        out = dict(ins)
        box = ins.get("box_xyxy_px")
        if box is not None:
            out["box_xyxy_px"] = [
                float(box[0]) * sx,
                float(box[1]) * sy,
                float(box[2]) * sx,
                float(box[3]) * sy,
            ]

        keypoints = ins.get("keypoints")
        if keypoints is not None:
            kp_out = {}
            for name, record in keypoints.items():
                rec = dict(record)
                xy = rec.get("xy_px")
                if xy is not None:
                    rec["xy_px"] = [float(xy[0]) * sx, float(xy[1]) * sy]
                kp_out[name] = rec
            out["keypoints"] = kp_out

        mask_id = int(out["mask_id"])
        out["visible_area_px"] = int(np.count_nonzero(resized == mask_id))
        scaled.append(out)
    return resized.astype(np.uint16, copy=False), scaled


def _put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.38,
    thickness: int = 1,
    fg: tuple[int, int, int] = (255, 255, 255),
    bg: tuple[int, int, int] = (20, 20, 20),
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    x = max(0, min(x, image.shape[1] - tw - 5))
    y = max(th + 4, min(y, image.shape[0] - baseline - 2))
    cv2.rectangle(image, (x - 2, y - th - 3), (x + tw + 3, y + baseline + 2), bg, -1)
    cv2.putText(image, text, (x, y), font, scale, fg, thickness, cv2.LINE_AA)


def draw_prediction_overlay(
    frame_bgr: np.ndarray,
    instance_mask: np.ndarray,
    instances: list[dict],
    runtime: dict,
    frame_index: int,
    source_fps: float,
    alpha: float = 0.36,
    draw_keypoints: bool = True,
) -> np.ndarray:
    """Draw distinct translucent instance masks and compact inference metrics."""
    vis = frame_bgr.copy()
    overlay = frame_bgr.copy()

    for ins in instances:
        mask_id = int(ins["mask_id"])
        pix = instance_mask == mask_id
        if not np.any(pix):
            continue
        color = instance_color(ins)
        overlay[pix] = color

    cv2.addWeighted(overlay, float(alpha), vis, 1.0 - float(alpha), 0.0, vis)

    for ins in instances:
        mask_id = int(ins["mask_id"])
        pix_u8 = (instance_mask == mask_id).astype(np.uint8) * 255
        if not np.any(pix_u8):
            continue
        color = instance_color(ins)
        contours, _ = cv2.findContours(pix_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(vis, contours, -1, color, 1, cv2.LINE_AA)

        box = ins.get("box_xyxy_px")
        if box is not None:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        else:
            ys, xs = np.where(pix_u8 > 0)
            x1, y1 = int(xs.min()), int(ys.min())

        if draw_keypoints and ins.get("keypoints"):
            for kp in ins["keypoints"].values():
                xy = kp.get("xy_px")
                if xy is None:
                    continue
                x, y = int(round(xy[0])), int(round(xy[1]))
                if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
                    cv2.circle(vis, (x, y), 2, color, -1, cv2.LINE_AA)

        track = ins.get("track_id") or f"mask_{mask_id}"
        conf = ins.get("detection_score")
        depth = (ins.get("pose") or {}).get("depth_camera_z_m")
        parts = [str(track)]
        if conf is not None:
            parts.append(f"conf {float(conf):.2f}")
        if depth is not None and math.isfinite(float(depth)):
            parts.append(f"z {float(depth):.2f}m")
        _put_text(vis, " | ".join(parts), (x1 + 2, max(14, y1 - 4)), scale=0.34)

    confidences = [float(x["detection_score"]) for x in instances if x.get("detection_score") is not None]
    mean_conf = float(np.mean(confidences)) if confidences else 0.0
    infer_ms = float(runtime.get("inference_ms", 0.0))
    post_ms = float(runtime.get("postprocess_ms", 0.0))
    total_ms = float(runtime.get("total_ms", infer_ms + post_ms))
    model_fps = 1000.0 / total_ms if total_ms > 0 else 0.0

    hud = [
        f"frame {frame_index} | gates {len(instances)} | mean conf {mean_conf:.2f}",
        f"infer {infer_ms:.1f} ms | post {post_ms:.1f} ms | total {total_ms:.1f} ms | {model_fps:.1f} FPS",
        f"source {source_fps:.1f} FPS",
    ]
    y = 15
    for line in hud:
        _put_text(vis, line, (6, y), scale=0.34)
        y += 15
    return vis
