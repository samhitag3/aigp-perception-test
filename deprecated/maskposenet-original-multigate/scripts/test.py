#!/usr/bin/env python3
"""Evaluate a trained GateNet checkpoint on a dataset (IoU / Dice / P / R).

Usage:
    python scripts/test.py --config configs/gatenet_a2rl.yaml \
        --checkpoint runs/gatenet_a2rl/best.pt --data data/test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatenet.augment import build_augment
from gatenet.dataset import GateSegDataset
from gatenet.metrics import seg_metrics
from gatenet.model import build_gatenet
from gatenet.utils import AverageMeter, load_checkpoint, load_config, pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True, help="root with images/ + masks/")
    ap.add_argument("--device", default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = pick_device(args.device)
    size = (cfg["data"]["height"], cfg["data"]["width"])

    ds = GateSegDataset(args.data, size, augment=build_augment(None, train=False),
                        mask_suffix=cfg["data"].get("mask_suffix", ""))
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                        num_workers=cfg["train"].get("num_workers", 4))

    model = build_gatenet(cfg["model"]).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    meters = {k: AverageMeter() for k in
              ("iou", "dice", "precision", "recall", "accuracy")}
    with torch.no_grad():
        for img, mask in loader:
            img, mask = img.to(device), mask.to(device)
            out = model(img)[0]
            m = seg_metrics(out, mask, threshold=args.threshold)
            for k in meters:
                meters[k].update(m[k], img.size(0))

    print(f"Evaluated {len(ds)} images from {args.data}")
    for k, v in meters.items():
        print(f"  {k:10s}: {v.avg:.4f}")


if __name__ == "__main__":
    main()
