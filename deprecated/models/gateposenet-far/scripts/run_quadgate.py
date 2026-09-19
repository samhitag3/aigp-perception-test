#!/usr/bin/env python3
"""Run every trained model's mask through QuAdGate corner detection.

For each model (GateNet, YOLO26[, YOLOE]) and each test frame:
  mask -> clean_mask -> QuAdGate (LSD edges -> line intersections) -> corner
  candidates -> 4-corner quad -> PnP -> orientation.

Produces a montage (input | GT | <model> mask + QuAdGate corners + pose ...) and
reports, per model, the QuAdGate detection rate and gate-normal error vs GT.

    python scripts/run_quadgate.py \
        --gatenet-config configs/gatenet_synth.yaml \
        --gatenet-ckpt runs/gatenet_synth/best.pt \
        --yolo-weights yolo26/gate6000-n320/weights/best.pt \
        --out docs/eval_quadgate.jpg
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
from perception.pose import (clean_mask, normal_angle_error_deg, order_quad,
                             pose_from_corners)
from perception.quadgate import QuAdGate
from scripts.eval_perception import GT_TO_CV, build_gt_lookup, gt_for, label, overlay

PURPLE = (220, 40, 180)
YELLOW = (0, 255, 255)


def quad_from_quadgate(quad_det: QuAdGate, mask: np.ndarray):
    """Run QuAdGate, turn its corner candidates into an ordered 4-corner quad."""
    cands = quad_det.candidates(mask)
    pts = np.array([c["xy"] for c in cands], np.float32) if cands else np.empty((0, 2), np.float32)
    quad = None
    if len(pts) >= 4:
        hull = cv2.convexHull(pts)
        peri = cv2.arcLength(hull, True)
        for k in np.linspace(0.02, 0.15, 12):
            approx = cv2.approxPolyDP(hull, k * peri, True)
            if len(approx) == 4:
                quad = order_quad(approx.reshape(-1, 2).astype(np.float32))
                break
    return pts, quad


def run_model_frame(mask, K, quad_det):
    """mask -> QuAdGate corners -> quad -> pose. Returns (pts, quad, pose)."""
    m = clean_mask(mask)
    pts, quad = quad_from_quadgate(quad_det, m)
    pose = pose_from_corners(quad, K, side=1.0) if quad is not None else None
    return pts, quad, pose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gatenet-config", required=True)
    ap.add_argument("--gatenet-ckpt", required=True)
    ap.add_argument("--yolo-weights", default=None)
    ap.add_argument("--yolo-family", default="yolo26")
    ap.add_argument("--test-dir", default="data/test")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--n-metric", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="docs/eval_quadgate.jpg")
    args = ap.parse_args()

    cfg = load_config(args.gatenet_config)
    Hn, Wn = cfg["data"]["height"], cfg["data"]["width"]
    net = build_gatenet(cfg["model"])
    load_checkpoint(args.gatenet_ckpt, net, map_location=args.device)
    gnet = GateNetInference(net).to(args.device).eval()

    yolo = None
    if args.yolo_weights:
        from ultralytics import YOLO, YOLOE
        yolo = (YOLOE if args.yolo_family == "yoloe" else YOLO)(args.yolo_weights)

    quad_det = QuAdGate(threshold=0.5)
    pairs = load_test_pairs(args.test_dir)
    lookup = build_gt_lookup()
    K_default = next((K for _, (K, _) in lookup.items()), None)

    def gatenet_mask(bgr, shape):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(cv2.resize(rgb, (Wn, Hn))).permute(2, 0, 1).float()[None] / 255.0
        with torch.no_grad():
            prob = gnet(t.to(args.device))[0, 0].cpu().numpy()
        return (cv2.resize(prob, (shape[1], shape[0])) > 0.5).astype(np.uint8)

    def yolo_mask(ip, shape):
        out = np.zeros(shape, np.uint8)
        if yolo is None:
            return out
        r = yolo.predict(str(ip), imgsz=args.imgsz, conf=0.25, device=args.device, verbose=False)[0]
        if r.masks is not None:
            for m in r.masks.data.cpu().numpy():
                out |= (cv2.resize(m.astype(np.float32), (shape[1], shape[0])) > 0.5).astype(np.uint8)
        return out

    models = [("GateNet", gatenet_mask, PURPLE)]
    if yolo is not None:
        models.append(("YOLO26", lambda ip, sh, _f=None: None, YELLOW))  # placeholder, handled below

    # ---- aggregate ------------------------------------------------------
    agg = {}
    midx = np.linspace(0, len(pairs) - 1, min(args.n_metric, len(pairs))).astype(int)
    for name in (["GateNet"] + (["YOLO26"] if yolo is not None else [])):
        dets, errs = 0, []
        for i in midx:
            stem, ip, mp = pairs[i]
            bgr = cv2.imread(str(ip)); gt = read_gt_mask(mp)
            K, gobj = gt_for(stem, lookup); K = K if K is not None else K_default
            mask = gatenet_mask(bgr, gt.shape) if name == "GateNet" else yolo_mask(ip, gt.shape)
            _, quad, pose = run_model_frame(mask, K, quad_det)
            if pose is not None:
                dets += 1
                if gobj and "normal_cam" in gobj:
                    errs.append(normal_angle_error_deg(pose.normal_cam, GT_TO_CV * np.array(gobj["normal_cam"])))
        agg[name] = {
            "quadgate_detect_rate": round(dets / len(midx), 4),
            "normal_err_deg_median": round(float(np.median(errs)), 3) if errs else None,
            "normal_err_deg_mean": round(float(np.mean(errs)), 3) if errs else None,
            "frames": int(len(midx)),
        }
    print("QuAdGate eval:", json.dumps(agg, indent=2))

    # ---- montage --------------------------------------------------------
    rows = []
    for i in np.linspace(0, len(pairs) - 1, args.n).astype(int):
        stem, ip, mp = pairs[i]
        bgr = cv2.imread(str(ip)); gt = read_gt_mask(mp)
        K, gobj = gt_for(stem, lookup); K = K if K is not None else K_default
        panels = [label(bgr, "input"), label(cv2.cvtColor(gt * 255, cv2.COLOR_GRAY2BGR), "GT")]
        for name, color in ([("GateNet", PURPLE)] + ([("YOLO26", YELLOW)] if yolo is not None else [])):
            mask = gatenet_mask(bgr, gt.shape) if name == "GateNet" else yolo_mask(ip, gt.shape)
            cm = clean_mask(mask)
            pts, quad, pose = run_model_frame(mask, K, quad_det)
            vis = overlay(bgr, cm, color, alpha=0.45)
            for (x, y) in pts.astype(int):                       # QuAdGate candidates
                cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
            txt = f"{name}: {len(pts)} corners"
            if quad is not None:
                cv2.polylines(vis, [quad.astype(int)], True, (255, 255, 0), 2)
            if pose is not None:
                cv2.drawFrameAxes(vis, K, None, pose.rvec, pose.tvec, 0.5, 2)
                if gobj and "normal_cam" in gobj:
                    e = normal_angle_error_deg(pose.normal_cam, GT_TO_CV * np.array(gobj["normal_cam"]))
                    txt += f"  dN {e:.0f}"
            panels.append(label(vis, txt, color))
        rows.append(cv2.hconcat(panels))

    montage = cv2.vconcat(rows)
    if montage.shape[1] > 1900:
        s = 1900 / montage.shape[1]
        montage = cv2.resize(montage, (1900, int(montage.shape[0] * s)), interpolation=cv2.INTER_AREA)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), montage, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {out}")
    Path("docs/results").mkdir(parents=True, exist_ok=True)
    Path("docs/results/eval_quadgate.json").write_text(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
