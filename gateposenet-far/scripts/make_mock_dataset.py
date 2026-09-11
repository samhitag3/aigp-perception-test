#!/usr/bin/env python3
"""Generate a small dataset in the sam3-autolabeler OUTPUT format.

Useful as (a) a format reference, (b) a CI / offline test fixture so the import
-> train -> eval pipeline can be exercised without the (transient) autolabeler
datasets present. Produces, under <out>:

    images/frame_XXXXX.png      RGB gate-on-background
    masks/frame_XXXXX.png       binary gate-frame mask (0/255)
    labels/frame_XXXXX.txt      YOLO seg polygon, class 0 (from mask contour)
    data.yaml                   nc:1 names:['gate']
    ground_truth.json           intrinsics + per-frame bbox + corners

    python scripts/make_mock_dataset.py --out /tmp/mock_gate --n 60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def warp_gate(size, rng):
    """Draw an orange square gate frame, perspective+rotate it, return (rgb_gate,
    alpha, outer_corners)."""
    H, W = size
    canvas = np.zeros((H, W, 3), np.uint8)
    alpha = np.zeros((H, W), np.uint8)
    s = int(rng.uniform(0.3, 0.8) * min(H, W))
    th = max(6, s // 12)
    cx, cy = W // 2, H // 2
    a, b = cx - s // 2, cx + s // 2
    c, d = cy - s // 2, cy + s // 2
    cv2.rectangle(canvas, (a, c), (b, d), (255, 110, 0), th)
    cv2.rectangle(alpha, (a, c), (b, d), 255, th)
    corners = np.float32([[a, c], [b, c], [b, d], [a, d]])  # TL,TR,BR,BL

    # perspective + rotation
    jit = 0.12
    dst = corners_full = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    off = rng.uniform(-1, 1, (4, 2)) * np.array([jit * W, jit * H])
    M = cv2.getPerspectiveTransform(src, (src + off).astype(np.float32))
    canvas = cv2.warpPerspective(canvas, M, (W, H))
    alpha = cv2.warpPerspective(alpha, M, (W, H))
    cor = cv2.perspectiveTransform(corners[None], M)[0]
    ang = rng.uniform(-180, 180)
    R = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
    canvas = cv2.warpAffine(canvas, R, (W, H))
    alpha = cv2.warpAffine(alpha, R, (W, H))
    cor = (R[:, :2] @ cor.T).T + R[:, 2]
    return canvas, alpha, cor


def yolo_polygon(mask):
    """Largest external contour -> normalized 'cls x1 y1 ...' YOLO seg line."""
    H, W = mask.shape
    cnts, _ = cv2.findContours((mask > 127).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    eps = 0.004 * cv2.arcLength(c, True)
    poly = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    if len(poly) < 3:
        return None
    pts = " ".join(f"{x/W:.6f} {y/H:.6f}" for x, y in poly)
    return f"0 {pts}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    for sub in ("images", "masks", "labels"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    H, W = args.height, args.width
    rng = np.random.default_rng(args.seed)

    gt = {"corner_order": ["TL", "TR", "BR", "BL"],
          "intrinsics": {"fx": 0.5 * W, "fy": 0.5 * W, "cx": 0.5 * W, "cy": 0.5 * H,
                         "width": W, "height": H},
          "objects": []}

    for i in range(args.n):
        bg = np.tile(rng.integers(40, 180, 3).astype(np.uint8), (H, W, 1))
        for _ in range(40):
            p1 = tuple(rng.integers(0, [W, H], 2).tolist())
            p2 = tuple(rng.integers(0, [W, H], 2).tolist())
            cv2.line(bg, p1, p2, rng.integers(0, 255, 3).tolist(), int(rng.integers(1, 4)))
        gate, alpha, corners = warp_gate((H, W), rng)
        a3 = (alpha[..., None] > 127).astype(np.float32)
        img = (gate * a3 + bg * (1 - a3)).astype(np.uint8)
        img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
        mask = ((alpha > 127).astype(np.uint8)) * 255

        stem = f"frame_{i:05d}"
        cv2.imwrite(str(out / "images" / f"{stem}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / "masks" / f"{stem}.png"), mask)
        line = yolo_polygon(mask)
        (out / "labels" / f"{stem}.txt").write_text((line or "") + "\n")
        xs, ys = corners[:, 0], corners[:, 1]
        gt["objects"].append({
            "frame": i, "file": f"{stem}.png", "obj_id": 0, "label": "gate",
            "bbox": [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
            "corners": corners.tolist(),
        })

    (out / "data.yaml").write_text(
        "path: .\ntrain: images\nval: images\nnc: 1\nnames: ['gate']\n")
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=2))
    print(f"wrote {args.n} frames -> {out} (images/masks/labels + data.yaml + ground_truth.json)")


if __name__ == "__main__":
    main()
