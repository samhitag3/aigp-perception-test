#!/usr/bin/env python3
"""Proof-of-concept: build a tiny synthetic gate dataset, train GateNet briefly,
and emit visual proof (input / GT / prediction / overlay / QuAdGate corners).

Outputs go to docs/poc/. This validates the whole stack end-to-end on CPU.

    python scripts/make_poc.py --epochs 25 --res 128 --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatenet.augment import Augment
from gatenet.losses import DeepSupervisionLoss
from gatenet.metrics import seg_metrics
from gatenet.model import GateNetInference, build_gatenet
from gatenet.schedule import build_scheduler
from gatenet.synth import SyntheticGateDataset
from gatenet.utils import AverageMeter, set_seed
from perception.quadgate import QuAdGate


def make_gate_template(path: Path, size=400, thickness=34):
    """An orange square gate frame on a transparent background (RGBA)."""
    img = np.zeros((size, size, 4), np.uint8)
    a, b = thickness, size - thickness
    # outer orange frame
    cv2.rectangle(img, (a, a), (b, b), (0, 110, 255, 255), thickness)  # BGRA-ish
    # set RGB to orange and alpha where drawn
    rgba = img.copy()
    cv2.imwrite(str(path), rgba)


def make_backgrounds(d: Path, n=4, size=512):
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    for i in range(n):
        base = rng.integers(40, 180, size=3)
        bg = np.tile(base.astype(np.uint8), (size, size, 1))
        # add some texture / clutter so it's not trivial
        for _ in range(60):
            p1 = tuple(rng.integers(0, size, 2).tolist())
            p2 = tuple(rng.integers(0, size, 2).tolist())
            col = rng.integers(0, 255, 3).tolist()
            cv2.line(bg, p1, p2, col, int(rng.integers(1, 4)))
        cv2.imwrite(str(d / f"bg_{i}.jpg"), bg)


def to_bgr(img_t):
    img = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--train-size", type=int, default=192)
    ap.add_argument("--out", default="docs/poc")
    args = ap.parse_args()

    set_seed(0)
    device = torch.device(args.device)
    out = Path(args.out)
    (out).mkdir(parents=True, exist_ok=True)
    assets = out / "_assets"
    assets.mkdir(exist_ok=True)

    # 1. synthetic data --------------------------------------------------
    gate_dir = assets / "gates"; gate_dir.mkdir(exist_ok=True)
    bg_dir = assets / "backgrounds"
    make_gate_template(gate_dir / "gate.png")
    make_backgrounds(bg_dir)
    size = (args.res, args.res)

    train_aug = Augment(train=True, rotate_deg=10, motion_blur_p=0.3,
                        gauss_noise_p=0.3, gauss_noise_std=8.0)
    train_ds = SyntheticGateDataset(gate_dir, bg_dir, size, length=args.train_size,
                                    augment=train_aug, alpha_from="alpha",
                                    seed=1, perspective=0.12)
    val_ds = SyntheticGateDataset(gate_dir, bg_dir, size, length=32,
                                  augment=None, alpha_from="alpha",
                                  seed=999, perspective=0.12)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    # 2. model / train ---------------------------------------------------
    model = build_gatenet({"width_factor": 2}).to(device)
    print(f"GateNet f=2: {model.num_parameters()/1e6:.3f} M params on {device}")
    crit = DeepSupervisionLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sch = build_scheduler(opt, {"milestones": [int(args.epochs*0.5), int(args.epochs*0.8)]})

    curve = []
    for ep in range(args.epochs):
        model.train()
        lm = AverageMeter()
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            opt.zero_grad(set_to_none=True)
            loss, _ = crit(model(img), mask)
            loss.backward(); opt.step()
            lm.update(loss.item(), img.size(0))
        sch.step()
        model.eval()
        im = AverageMeter()
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(device), mask.to(device)
                im.update(seg_metrics(model(img)[0], mask)["iou"], img.size(0))
        curve.append(im.avg)
        print(f"  epoch {ep+1:2d}/{args.epochs}  loss={lm.avg:.3f}  val_IoU={im.avg:.3f}")

    # 3. proof images ----------------------------------------------------
    infer = GateNetInference(model).to(device).eval()
    quad = QuAdGate(threshold=0.5)
    panels = []
    with torch.no_grad():
        for k in range(4):
            img_t, gt_t = val_ds[k * 7]
            bgr = to_bgr(img_t)
            prob = infer(img_t[None].to(device))[0, 0].cpu().numpy()
            pred = (prob > 0.5).astype(np.uint8)
            gt = (gt_t[0].numpy() * 255).astype(np.uint8)

            pred_vis = cv2.cvtColor(pred * 255, cv2.COLOR_GRAY2BGR)
            overlay = bgr.copy()
            overlay[pred > 0] = (0.45 * overlay[pred > 0] +
                                 0.55 * np.array([0, 0, 255])).astype(np.uint8)
            corners = overlay.copy()
            for c in quad.candidates(pred):
                x, y = c["xy"].astype(int)
                cv2.circle(corners, (x, y), 4, (0, 255, 0), -1)

            row = cv2.hconcat([
                label(bgr, "input"),
                label(cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR), "ground truth"),
                label(pred_vis, "GateNet pred"),
                label(overlay, "overlay"),
                label(corners, "QuAdGate corners"),
            ])
            panels.append(row)
            cv2.imwrite(str(out / f"sample_{k}.png"), row)

    montage = cv2.vconcat(panels)
    cv2.imwrite(str(out / "poc_montage.png"), montage)

    # training curve
    cw = 600; ch = 240; plot = np.full((ch, cw, 3), 255, np.uint8)
    if len(curve) > 1:
        for i in range(1, len(curve)):
            x0 = int((i-1)/(len(curve)-1)*(cw-40))+20
            x1 = int(i/(len(curve)-1)*(cw-40))+20
            y0 = ch-20-int(curve[i-1]*(ch-40))
            y1 = ch-20-int(curve[i]*(ch-40))
            cv2.line(plot, (x0, y0), (x1, y1), (200, 80, 0), 2)
    cv2.putText(plot, f"val IoU vs epoch (final={curve[-1]:.3f})", (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out / "training_curve.png"), plot)

    print(f"\nfinal val IoU = {curve[-1]:.3f}")
    print(f"wrote proof images -> {out}/  (poc_montage.png, sample_*.png, training_curve.png)")


if __name__ == "__main__":
    main()
