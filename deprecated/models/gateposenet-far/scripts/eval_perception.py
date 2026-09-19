#!/usr/bin/env python3
"""POC-style perception eval for GateNet: mask -> corners -> ORIENTATION (PnP).

Per test frame, renders a row:
  input | GT mask | GateNet (purple, denoised) | corners+quad | pose (axes + orientation)

and, where ground-truth is available (sam3-autolabeler `ground_truth.json`),
reports the gate-normal orientation error vs GT and an aggregate over the set.

    python scripts/eval_perception.py \
        --gatenet-config configs/gatenet_synth.yaml \
        --gatenet-ckpt runs/gatenet_synth/best.pt \
        --out docs/eval_perception.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.common import load_test_pairs, read_gt_mask
from gatenet.model import GateNetInference, build_gatenet
from gatenet.utils import load_checkpoint, load_config
from perception.pose import (clean_mask, draw_pose, gate_pose_from_mask,
                             normal_angle_error_deg)

PURPLE = (220, 40, 180)   # BGR — high-contrast vs the reddish backgrounds
GT_TO_CV = np.array([-1.0, -1.0, 1.0])   # GT normal_cam uses flipped x,y vs OpenCV


def label(img, text, color=(255, 255, 255)):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return out


def overlay(bgr, mask, color, alpha=0.55):
    m = mask.astype(bool); out = bgr.copy()
    out[m] = (alpha * np.array(color) + (1 - alpha) * out[m]).astype(np.uint8)
    return out


def build_gt_lookup():
    """{dataset: (K, {file: obj})} from the import manifest + ground_truth.json."""
    man = Path("data/meta/import_manifest.json")
    if not man.exists():
        return {}
    out = {}
    for ds in json.loads(man.read_text()).get("datasets", []):
        gt = Path(ds["path"]) / "ground_truth.json"
        if not gt.exists():
            continue
        d = json.loads(gt.read_text())
        intr = d.get("intrinsics", {})
        K = np.array(intr["K"], float) if "K" in intr else None
        by_file = {o["file"]: o for o in d.get("objects", [])}
        out[ds["name"]] = (K, by_file)
    return out


def gt_for(stem, lookup):
    """Map test stem '<dataset>__frame_xxxxx' -> (K, gt_obj)."""
    if "__" not in stem:
        return None, None
    name, frame = stem.split("__", 1)
    if name not in lookup:
        return None, None
    K, by_file = lookup[name]
    return K, by_file.get(frame + ".png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gatenet-config", required=True)
    ap.add_argument("--gatenet-ckpt", required=True)
    ap.add_argument("--test-dir", default="data/test")
    ap.add_argument("--n", type=int, default=6, help="frames in the montage")
    ap.add_argument("--n-metric", type=int, default=150, help="frames for the aggregate")
    ap.add_argument("--side", type=float, default=1.0, help="gate side (units); orientation is scale-invariant")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="docs/eval_perception.jpg")
    args = ap.parse_args()

    cfg = load_config(args.gatenet_config)
    Hn, Wn = cfg["data"]["height"], cfg["data"]["width"]
    net = build_gatenet(cfg["model"])
    load_checkpoint(args.gatenet_ckpt, net, map_location=args.device)
    gnet = GateNetInference(net).to(args.device).eval()

    pairs = load_test_pairs(args.test_dir)
    lookup = build_gt_lookup()
    K_default = None
    for _, (K, _) in lookup.items():
        K_default = K; break

    def predict_mask(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(cv2.resize(rgb, (Wn, Hn))).permute(2, 0, 1).float()[None] / 255.0
        with torch.no_grad():
            prob = gnet(t.to(args.device))[0, 0].cpu().numpy()
        return prob

    # ---- aggregate orientation error -------------------------------------
    errs, dets = [], 0
    midx = np.linspace(0, len(pairs) - 1, min(args.n_metric, len(pairs))).astype(int)
    for i in midx:
        stem, ip, mp = pairs[i]
        bgr = cv2.imread(str(ip)); gt = read_gt_mask(mp)
        prob = cv2.resize(predict_mask(bgr), (gt.shape[1], gt.shape[0]))
        K, gobj = gt_for(stem, lookup)
        K = K if K is not None else K_default
        pose, _ = gate_pose_from_mask((prob > 0.5).astype(np.uint8), K, side=args.side)
        if pose is None:
            continue
        dets += 1
        if gobj and "normal_cam" in gobj:
            errs.append(normal_angle_error_deg(pose.normal_cam, GT_TO_CV * np.array(gobj["normal_cam"])))
    agg = {
        "frames": int(len(midx)),
        "gate_detected_rate": round(dets / max(1, len(midx)), 4),
        "normal_err_deg_mean": round(float(np.mean(errs)), 3) if errs else None,
        "normal_err_deg_median": round(float(np.median(errs)), 3) if errs else None,
    }
    print("orientation eval:", json.dumps(agg))

    # ---- montage ---------------------------------------------------------
    rows = []
    for i in np.linspace(0, len(pairs) - 1, args.n).astype(int):
        stem, ip, mp = pairs[i]
        bgr = cv2.imread(str(ip)); gt = read_gt_mask(mp)
        prob = cv2.resize(predict_mask(bgr), (gt.shape[1], gt.shape[0]))
        pred = clean_mask((prob > 0.5).astype(np.uint8))
        K, gobj = gt_for(stem, lookup); K = K if K is not None else K_default

        gt_vis = cv2.cvtColor(gt * 255, cv2.COLOR_GRAY2BGR)
        gn_vis = overlay(bgr, pred, PURPLE)

        pose, quad = gate_pose_from_mask(pred, K, side=args.side)
        corner_vis = bgr.copy()
        pose_vis = bgr.copy()
        txt = "no gate"
        if pose is not None:
            cv2.polylines(corner_vis, [pose.corners.astype(int)], True, (0, 255, 0), 2)
            for (x, y) in pose.corners.astype(int):
                cv2.circle(corner_vis, (int(x), int(y)), 5, (0, 255, 255), -1)
            pose_vis = draw_pose(bgr, pose, K, side=args.side)
            r, p, yw = pose.rpy_deg
            txt = f"yaw {yw:.0f} pitch {p:.0f}"
            if gobj and "normal_cam" in gobj:
                e = normal_angle_error_deg(pose.normal_cam, GT_TO_CV * np.array(gobj["normal_cam"]))
                txt += f"  dNormal {e:.1f}deg"
        rows.append(cv2.hconcat([
            label(bgr, "input"),
            label(gt_vis, "ground truth"),
            label(gn_vis, "GateNet mask"),
            label(corner_vis, "corners"),
            label(pose_vis, "orientation: " + txt, (0, 255, 255)),
        ]))

    montage = cv2.vconcat(rows)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    # save a reasonably-sized jpg
    if montage.shape[1] > 1800:
        s = 1800 / montage.shape[1]
        montage = cv2.resize(montage, (1800, int(montage.shape[0] * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out), montage, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {out}")
    Path("docs/results").mkdir(parents=True, exist_ok=True)
    Path("docs/results/eval_orientation.json").write_text(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
