#!/usr/bin/env python3
"""Estimate gate position from images: camera frame + downstream BODY-NED.

The deployment pipeline in one script:

    image(s) --(GatePoseNet, camera-frame)--> position_cam, R_cam_gate
             --(perception.body_frame + --camera-tilt)--> position_body

    uv run python scripts/estimate_position.py --images-dir data/test_images \
        --camera-tilt 30

Files are processed in sorted order as one temporal sequence (--independent
resets state per image). Also supports the classical fallback: --use-pnp
solves position from the model's predicted corners via IPPE PnP instead of
the direct metric head (needs the camera intrinsics; --fx/--cx/... override
the AIGP nominals).

Output: one line per frame + estimates.json (camera frame ALWAYS; body frame
is derived downstream with the mount tilt you pass — the model itself is
mount-agnostic).
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
from perception.body_frame import (DEFAULT_MOUNT_TILT_DEG,
                                   MOUNT_TILT_DETENTS_DEG, gate_pose_to_body)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
AIGP_GATE_OUTER_M = 2.7


def pnp_position(corners_px, K, gate_m=AIGP_GATE_OUTER_M):
    h = gate_m / 2.0
    obj = np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]],
                   dtype=np.float64)
    ok, _, tvec = cv2.solvePnP(obj, np.asarray(corners_px, dtype=np.float64),
                               K, None, flags=cv2.SOLVEPNP_IPPE)
    return tvec.reshape(3) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gateposenet_traj.yaml")
    ap.add_argument("--checkpoint", default="runs/gateposenet_traj/best.pt")
    ap.add_argument("--images-dir", default="data/test_images")
    ap.add_argument("--out", default="runs/position_estimates")
    ap.add_argument("--camera-tilt", type=float, default=DEFAULT_MOUNT_TILT_DEG,
                    help=f"mount up-tilt on THIS frame (detents "
                         f"{MOUNT_TILT_DETENTS_DEG}; default 30)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--independent", action="store_true")
    ap.add_argument("--use-pnp", action="store_true",
                    help="position from predicted corners via IPPE PnP "
                         "(calibrated path) instead of the direct head")
    ap.add_argument("--fx", type=float, default=320.0)
    ap.add_argument("--fy", type=float, default=320.0)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = pick_device(args.device)
    d = cfg["data"]
    mh, mw = int(d.get("height", 192)), int(d.get("width", 320))

    paths = sorted(p for p in Path(args.images_dir).iterdir()
                   if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise SystemExit(f"no images in {args.images_dir}")

    model = build_gateposenet(cfg["model"]).to(device).eval()
    load_checkpoint(args.checkpoint, model, map_location=device)

    ego = torch.zeros(1, 7, device=device)
    ego[0, 6] = 1.0 / max(1.0, float(args.fps))
    h = None
    records = []
    print(f"{len(paths)} frames | tilt {args.camera_tilt} deg | "
          f"{'PnP' if args.use_pnp else 'direct'} position path")
    with torch.no_grad():
        for p in paths:
            raw = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if raw is None:
                continue
            H0, W0 = raw.shape[:2]
            rgb = cv2.cvtColor(cv2.resize(raw, (mw, mh),
                                          interpolation=cv2.INTER_AREA),
                               cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(rgb.transpose(2, 0, 1))[None].float() / 255.0
            if args.independent:
                h = None
            out, h = model.step(x.to(device), ego, h)

            vis = float(torch.sigmoid(out["visible_logit"])[0])
            R_cam = out["R"][0].cpu().numpy()
            if args.use_pnp:
                K = np.array([[args.fx, 0, args.cx or W0 / 2.0],
                              [0, args.fy, args.cy or H0 / 2.0],
                              [0, 0, 1.0]])
                corners = out["corners_uv"][0].cpu().numpy() * [W0, H0]
                p_cam = pnp_position(corners, K)
                if p_cam is None:
                    p_cam = out["position"][0].cpu().numpy()
            else:
                p_cam = out["position"][0].cpu().numpy()

            # PnP CHECK/CORRECT: the model is the estimator; the rigid-square
            # fit verifies its corner set (residual), cross-checks the two
            # position paths (agreement) and snaps corners onto the manifold.
            from perception.geometric_filter import filter_gate_estimate

            corners_all = out["corners_uv"][0].cpu().numpy() * [W0, H0]
            K_chk = np.array([[args.fx, 0, args.cx or W0 / 2.0],
                              [0, args.fy, args.cy or H0 / 2.0],
                              [0, 0, 1.0]])
            chk = filter_gate_estimate(corners_all, p_cam, K_chk,
                                       R_direct=R_cam)
            p_out = chk.position_fused

            body = gate_pose_to_body(p_out, R_cam, args.camera_tilt)
            rec = {
                "file": p.name,
                "visible_prob": vis,
                "position_cam_m": np.asarray(p_out).tolist(),
                "distance_m": body["distance_m"],
                "position_body_ned_m": body["position_body"].tolist(),
                "flythrough_axis_body": body["flythrough_axis_body"].tolist(),
                "camera_tilt_deg": float(args.camera_tilt),
                "pnp_check": {
                    "reproj_residual_px": (round(chk.reproj_residual_px, 2)
                                           if np.isfinite(
                                               chk.reproj_residual_px)
                                           else None),
                    "consistent": chk.consistent,
                    "agreement_m": (round(chk.agreement_m, 3)
                                    if np.isfinite(chk.agreement_m) else None),
                    "corners_snapped_px": (chk.corners_snapped.tolist()
                                           if chk.corners_snapped is not None
                                           else None),
                },
            }
            records.append(rec)
            pb = body["position_body"]
            print(f"  {p.name}: vis={vis:.2f} d={body['distance_m']:5.2f} m | "
                  f"cam [{p_cam[0]:+6.2f} {p_cam[1]:+6.2f} {p_cam[2]:+6.2f}] | "
                  f"body-NED [{pb[0]:+6.2f} {pb[1]:+6.2f} {pb[2]:+6.2f}]")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "estimates.json", "w") as f:
        json.dump({
            "note": ("position_cam is the model output (camera optical "
                     "frame, mount-agnostic); position_body_ned is derived "
                     "DOWNSTREAM via perception.body_frame with the mount "
                     "tilt passed at run time."),
            "frames": records,
        }, f, indent=2)
    print(f"wrote {out_dir / 'estimates.json'}")


if __name__ == "__main__":
    main()
