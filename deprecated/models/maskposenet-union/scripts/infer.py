#!/usr/bin/env python3
"""Run GateNet on a single image or a folder, saving mask overlays.

Usage:
    python scripts/infer.py --config configs/gatenet_a2rl.yaml \
        --checkpoint runs/gatenet_a2rl/best.pt --input img.jpg --output out/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatenet.dataset import IMG_EXTS
from gatenet.model import GateNetInference, build_gatenet
from gatenet.utils import load_checkpoint, load_config, pick_device


def preprocess(path, size):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = size
    rs = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(rs).permute(2, 0, 1).float()[None] / 255.0
    return img, t


def overlay(bgr, prob, threshold):
    mask = (cv2.resize(prob, (bgr.shape[1], bgr.shape[0])) > threshold)
    out = bgr.copy()
    out[mask] = (0.4 * out[mask] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="infer_out")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = pick_device(args.device)
    size = (cfg["data"]["height"], cfg["data"]["width"])
    thr = cfg.get("infer", {}).get("threshold", 0.5)

    net = build_gatenet(cfg["model"])
    load_checkpoint(args.checkpoint, net, map_location=device)
    model = GateNetInference(net).to(device).eval()

    inp = Path(args.input)
    paths = [inp] if inp.is_file() else sorted(
        p for p in inp.iterdir() if p.suffix.lower() in IMG_EXTS)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in paths:
        bgr, t = preprocess(p, size)
        with torch.no_grad():
            prob = model(t.to(device))[0, 0].cpu().numpy()
        cv2.imwrite(str(out_dir / f"{p.stem}_mask.png"), (prob * 255).astype(np.uint8))
        cv2.imwrite(str(out_dir / f"{p.stem}_overlay.png"), overlay(bgr, prob, thr))
        print(f"  {p.name} -> {out_dir}")


if __name__ == "__main__":
    main()
