#!/usr/bin/env python3
"""Render a side-by-side comparison of GateNet vs YOLO predictions on test images.

For each sampled test frame, builds a row:  input | ground truth | GateNet | YOLO
and stacks them into one montage. Used to produce docs/eval_comparison.jpg.

    python scripts/eval_visualize.py \
        --gatenet-config configs/gatenet_synth.yaml \
        --gatenet-ckpt runs/gatenet_synth/best.pt \
        --yolo-weights yolo26/gate6000-n320/weights/best.pt \
        --test-dir data/test --n 6 --out docs/eval_comparison.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.common import load_test_pairs, read_gt_mask
from gatenet.model import GateNetInference, build_gatenet
from gatenet.utils import load_checkpoint, load_config
from perception.pose import clean_mask

PURPLE = (220, 40, 180)   # BGR
YELLOW = (0, 255, 255)


def label(img, text, color=(255, 255, 255)):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return out


def overlay(bgr, mask, color):
    m = mask.astype(bool)
    out = bgr.copy()
    out[m] = (0.45 * out[m] + 0.55 * np.array(color)).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gatenet-config", required=True)
    ap.add_argument("--gatenet-ckpt", required=True)
    ap.add_argument("--yolo-weights", required=True)
    ap.add_argument("--yolo-family", default="yolo26")
    ap.add_argument("--test-dir", default="data/test")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="docs/eval_comparison.jpg")
    args = ap.parse_args()

    # GateNet
    cfg = load_config(args.gatenet_config)
    Hn, Wn = cfg["data"]["height"], cfg["data"]["width"]
    net = build_gatenet(cfg["model"])
    load_checkpoint(args.gatenet_ckpt, net, map_location=args.device)
    gnet = GateNetInference(net).to(args.device).eval()

    # YOLO
    from ultralytics import YOLO, YOLOE
    yolo = (YOLOE if args.yolo_family == "yoloe" else YOLO)(args.yolo_weights)

    pairs = load_test_pairs(args.test_dir)
    idxs = np.linspace(0, len(pairs) - 1, args.n).astype(int)

    rows = []
    for i in idxs:
        _, ip, mp = pairs[i]
        bgr = cv2.imread(str(ip))
        gt = read_gt_mask(mp)
        H, W = gt.shape

        # GateNet mask
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(cv2.resize(rgb, (Wn, Hn))).permute(2, 0, 1).float()[None] / 255.0
        with torch.no_grad():
            prob = gnet(t.to(args.device))[0, 0].cpu().numpy()
        gn_mask = clean_mask((cv2.resize(prob, (W, H)) > 0.5).astype(np.uint8))

        # YOLO union mask
        r = yolo.predict(str(ip), imgsz=args.imgsz, conf=0.25, device=args.device,
                         verbose=False)[0]
        yo_mask = np.zeros((H, W), np.uint8)
        if r.masks is not None:
            for m in r.masks.data.cpu().numpy():
                yo_mask |= (cv2.resize(m.astype(np.float32), (W, H)) > 0.5).astype(np.uint8)

        gt_vis = cv2.cvtColor(gt * 255, cv2.COLOR_GRAY2BGR)
        row = cv2.hconcat([
            label(bgr, "input"),
            label(gt_vis, "ground truth"),
            label(overlay(bgr, gn_mask, PURPLE), "GateNet"),
            label(overlay(bgr, yo_mask, YELLOW), "YOLO26"),
        ])
        rows.append(row)

    montage = cv2.vconcat(rows)
    if montage.shape[1] > 1800:
        s = 1800 / montage.shape[1]
        montage = cv2.resize(montage, (1800, int(montage.shape[0] * s)),
                             interpolation=cv2.INTER_AREA)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), montage, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {out}  ({montage.shape[1]}x{montage.shape[0]})")


if __name__ == "__main__":
    main()
