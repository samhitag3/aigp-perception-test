#!/usr/bin/env python3
"""Run the model over REAL footage (no GT) with unsupervised health metrics.

Real frames have no ground truth, but the system still self-reports health:

* visibility confidence over time (does it detect gates when they're there);
* the PnP CHECK — rigid-square reprojection residual of the predicted
  corners (geometric possibility) and PnP-vs-direct position agreement
  (cross-path consistency) — both computable WITHOUT labels;
* temporal smoothness — frame-to-frame jumps of the predicted position at
  30 Hz (a real gate does not teleport).

Outputs: annotated video, health_metrics.json, and a printed summary.

    uv run python scripts/run_real_footage.py --frames <dir> [--fps 30]
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

IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gateposenet_traj.yaml")
    ap.add_argument("--checkpoint", default="runs/gateposenet_traj/best.pt")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", default="runs/real_footage")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--fx", type=float, default=320.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = pick_device(args.device)
    d = cfg["data"]
    mh, mw = int(d.get("height", 192)), int(d.get("width", 320))

    paths = sorted(p for p in Path(args.frames).iterdir()
                   if p.suffix.lower() in IMAGE_EXTS)
    if args.max_frames:
        paths = paths[:args.max_frames]
    if not paths:
        raise SystemExit(f"no frames in {args.frames}")

    model = build_gateposenet(cfg["model"]).to(device).eval()
    load_checkpoint(args.checkpoint, model, map_location=device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    first = cv2.imread(str(paths[0]))
    H0, W0 = first.shape[:2]
    K = np.array([[args.fx, 0, W0 / 2.0], [0, args.fx, H0 / 2.0],
                  [0, 0, 1.0]])
    vw = cv2.VideoWriter(str(out_dir / "annotated.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W0, H0))

    ego = torch.zeros(1, 7, device=device)
    ego[0, 6] = 1.0 / args.fps
    h = None
    hist = []
    prev_pos = None
    with torch.no_grad():
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            rgb = cv2.cvtColor(cv2.resize(img, (mw, mh),
                                          interpolation=cv2.INTER_AREA),
                               cv2.COLOR_BGR2RGB)
            x = (torch.from_numpy(rgb.transpose(2, 0, 1))[None].float()
                 / 255.0).to(device)
            out, h = model.step(x, ego, h)

            vis = float(torch.sigmoid(out["visible_logit"])[0])
            pos = out["position"][0].cpu().numpy()
            R = out["R"][0].cpu().numpy()
            corners = out["corners_uv"][0].cpu().numpy() * [W0, H0]
            chk = filter_gate_estimate(corners, pos, K, R_direct=R)
            jump = (float(np.linalg.norm(pos - prev_pos))
                    if prev_pos is not None else 0.0)
            prev_pos = pos
            hist.append(dict(
                file=p.name, vis=vis,
                d=float(np.linalg.norm(pos)),
                residual=(chk.reproj_residual_px
                          if np.isfinite(chk.reproj_residual_px) else None),
                agreement=(chk.agreement_m
                           if np.isfinite(chk.agreement_m) else None),
                consistent=chk.consistent, jump_m=jump))

            # annotate
            col = (0, 220, 0) if vis >= 0.5 else (0, 140, 255)
            if vis >= 0.5:
                pts = corners.astype(int)
                cv2.polylines(img, [pts.reshape(-1, 1, 2)], True, col, 2)
                if chk.corners_snapped is not None and chk.consistent:
                    sp = chk.corners_snapped.astype(int)
                    cv2.polylines(img, [sp.reshape(-1, 1, 2)], True,
                                  (255, 200, 0), 1)
            cv2.rectangle(img, (0, 0), (W0, 22), (0, 0, 0), -1)
            r_txt = (f"resid={chk.reproj_residual_px:5.1f}px"
                     if np.isfinite(chk.reproj_residual_px) else "resid=  n/a")
            cv2.putText(
                img, f"vis={vis:.2f}  d={np.linalg.norm(pos):5.2f}m  {r_txt}"
                     f"  {'OK' if chk.consistent else '--'}",
                (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
                cv2.LINE_AA)
            vw.write(img)
    vw.release()

    v = np.array([r["vis"] for r in hist])
    det = v >= 0.5
    resid = np.array([r["residual"] for r in hist
                      if r["residual"] is not None and r["vis"] >= 0.5])
    agree = np.array([r["agreement"] for r in hist
                      if r["agreement"] is not None and r["vis"] >= 0.5])
    cons = np.array([r["consistent"] for r in hist if r["vis"] >= 0.5])
    jumps = np.array([r["jump_m"] for r in hist[1:]
                      if r["vis"] >= 0.5])
    summary = {
        "frames": len(hist),
        "detection_rate": round(float(det.mean()), 3),
        "median_pred_distance_m": round(float(np.median(
            [r["d"] for r in hist if r["vis"] >= 0.5] or [np.nan])), 2),
        "pnp_residual_px": {"median": round(float(np.median(resid)), 1)
                            if len(resid) else None,
                            "p90": round(float(np.percentile(resid, 90)), 1)
                            if len(resid) else None},
        "corner_set_consistent_rate": round(float(cons.mean()), 3)
        if len(cons) else None,
        "pnp_direct_agreement_m_median": round(float(np.median(agree)), 2)
        if len(agree) else None,
        "temporal_jump_m_median": round(float(np.median(jumps)), 3)
        if len(jumps) else None,
        "temporal_jump_m_p95": round(float(np.percentile(jumps, 95)), 3)
        if len(jumps) else None,
    }
    with open(out_dir / "health_metrics.json", "w") as f:
        json.dump({"summary": summary, "frames": hist}, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"-> {out_dir}/annotated.mp4 + health_metrics.json")


if __name__ == "__main__":
    main()
