#!/usr/bin/env python3
"""End-to-end perception demo: image -> GateNet mask -> QuAdGate corners -> PnP.

Demonstrates how the perception front-end is wired. Real deployment feeds
QuAdGate the *priors* projected from the EKF state estimate; here we run
corner-candidate extraction without priors (no matching) unless a flight-plan /
pose is supplied.

Usage:
    python scripts/demo_perception.py --config configs/gatenet_a2rl.yaml \
        --checkpoint runs/gatenet_a2rl/best.pt --input frame.jpg --output demo.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatenet.model import GateNetInference, build_gatenet
from gatenet.utils import load_checkpoint, load_config, pick_device
from perception.quadgate import QuAdGate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="demo_perception.png")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = pick_device(args.device)
    size = (cfg["data"]["height"], cfg["data"]["width"])
    thr = cfg.get("infer", {}).get("threshold", 0.5)

    net = build_gatenet(cfg["model"])
    load_checkpoint(args.checkpoint, net, map_location=device)
    model = GateNetInference(net).to(device).eval()

    bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rs = cv2.resize(rgb, (size[1], size[0]))
    t = torch.from_numpy(rs).permute(2, 0, 1).float()[None].to(device) / 255.0
    with torch.no_grad():
        prob = model(t)[0, 0].cpu().numpy()

    mask = (prob > thr).astype(np.uint8)
    quad = QuAdGate(threshold=0.5)
    cands = quad.candidates(mask)
    print(f"detected {len(cands)} corner candidates")

    vis = cv2.resize(bgr, (size[1], size[0])).copy()
    vis[mask > 0] = (0.5 * vis[mask > 0] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    for c in cands:
        x, y = c["xy"].astype(int)
        cv2.circle(vis, (x, y), 4, (0, 255, 0), -1)
    cv2.imwrite(args.output, vis)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
