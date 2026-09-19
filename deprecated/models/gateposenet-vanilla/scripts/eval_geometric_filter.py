#!/usr/bin/env python3
"""Measure what the known-gate-geometry filter buys on a dataset.

Streams sequences through model.step and compares position error of:
  direct   — the network's metric head (current default)
  pnp_raw  — IPPE PnP on raw predicted corners
  fused    — perception.geometric_filter (residual-gated, agreement-gated)
plus the value of the consistency residual as a CONFIDENCE signal
(error conditioned on residual quartiles).

    uv run python scripts/eval_geometric_filter.py --dataset <dir>
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

from gatenet.utils import load_checkpoint, load_config, pick_device
from gateposenet.model_single import \
    build_gateposenet_single as build_gateposenet
from perception.geometric_filter import filter_gate_estimate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gateposenet_traj.yaml")
    ap.add_argument("--checkpoint", default="runs/gateposenet_traj/best.pt")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = pick_device(args.device)
    d = cfg["data"]
    mh, mw = int(d.get("height", 192)), int(d.get("width", 320))

    with open(Path(args.dataset) / "ground_truth.json") as f:
        gt = json.load(f)
    fps = float(gt.get("fps") or 30.0)
    frames = {}
    for o in gt["objects"]:
        it = o.get("is_target")
        if (it is True) or (it is None and int(o.get("obj_id", 0)) == 0):
            frames[o["frame"]] = o
    seqs: dict = {}
    for fid, o in sorted(frames.items()):
        seqs.setdefault(o.get("seq_id", fid), []).append(o)

    model = build_gateposenet(cfg["model"]).to(device).eval()
    load_checkpoint(args.checkpoint, model, map_location=device)

    res = {"direct": [], "pnp_raw": [], "fused": []}
    residuals, errs_by_resid = [], []
    n_used = n_consistent = n_vis = 0
    with torch.no_grad():
        for objs in seqs.values():
            objs.sort(key=lambda o: (o.get("seq_t") or 0))
            h = None
            for o in objs:
                img = cv2.imread(str(Path(args.dataset) / "images" / o["file"]))
                if img is None:
                    continue
                H0, W0 = img.shape[:2]
                rgb = cv2.cvtColor(cv2.resize(img, (mw, mh),
                                              interpolation=cv2.INTER_AREA),
                                   cv2.COLOR_BGR2RGB)
                x = (torch.from_numpy(rgb.transpose(2, 0, 1))[None].float()
                     / 255.0).to(device)
                ego6 = o.get("ego_motion_cam") or [0.0] * 6
                ego = torch.tensor([[*ego6, 1.0 / fps]], dtype=torch.float32,
                                   device=device)
                out, h = model.step(x, ego, h)
                if not o.get("visible"):
                    continue
                n_vis += 1
                gt_pos = np.asarray(o["position_cam"])
                p_dir = out["position"][0].cpu().numpy()
                corners = out["corners_uv"][0].cpu().numpy() * [W0, H0]
                K = np.asarray(o["K"]) if o.get("K") is not None else \
                    np.array([[320, 0, W0 / 2], [0, 320, H0 / 2], [0, 0, 1.0]])
                est = filter_gate_estimate(corners, p_dir, K)
                res["direct"].append(np.linalg.norm(p_dir - gt_pos))
                if est.position_pnp is not None:
                    res["pnp_raw"].append(
                        np.linalg.norm(est.position_pnp - gt_pos))
                res["fused"].append(
                    np.linalg.norm(est.position_fused - gt_pos))
                residuals.append(est.reproj_residual_px)
                errs_by_resid.append((est.reproj_residual_px,
                                      float(np.linalg.norm(p_dir - gt_pos))))
                n_used += est.used_pnp
                n_consistent += est.consistent

    print(f"visible frames scored: {n_vis} | corner set consistent: "
          f"{n_consistent} ({n_consistent/max(1,n_vis):.0%}) | "
          f"PnP fused in: {n_used} ({n_used/max(1,n_vis):.0%})")
    for k, v in res.items():
        v = np.asarray(v)
        print(f"  {k:8s}: mean {v.mean():.3f} m | median "
              f"{np.median(v):.3f} m  (n={len(v)})")
    # residual as confidence: direct-head error by residual quartile
    eb = sorted(errs_by_resid)
    qs = np.array_split(np.asarray([e for _, e in eb]), 4)
    rq = np.array_split(np.asarray([r for r, _ in eb]), 4)
    print("  residual as confidence (direct-head pos err by residual quartile):")
    for i, (r, e) in enumerate(zip(rq, qs)):
        print(f"    Q{i+1} residual<{r.max():6.1f}px: median err "
              f"{np.median(e):.3f} m")


if __name__ == "__main__":
    main()
