#!/usr/bin/env python3
"""Run a trained GatePoseNet on a folder of test images.

Drop images into ``data/test_images/`` (any size — they are resized to the
model input; predictions are reported in the ORIGINAL pixel coordinates) and:

    uv run python scripts/run_test_images.py
    uv run python scripts/run_test_images.py --images-dir my_frames --fps 30

Files are processed in sorted filename order as ONE temporal sequence (the
ConvGRU hidden state carries across frames — exactly the 30 Hz deployment
loop). For unrelated single stills, pass ``--independent`` to reset the
recurrent state per image. No ego-motion is available for arbitrary images, so
the ego input is zeroed (the model is trained with ego-motion dropout and
degrades gracefully).

Writes to ``--out`` (default runs/test_images_out): an annotated copy of every
image and ``predictions.json`` with, per frame: corners_px, center_px,
position_cam (m), distance_m, R_cam_gate, fly-through axis, visibility.
Gate/camera constants come from configs/gate_aigp.yaml.
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

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gateposenet_traj.yaml")
    ap.add_argument("--gate-spec", default="configs/gate_aigp.yaml")
    ap.add_argument("--checkpoint", default="runs/gateposenet_traj/best.pt")
    ap.add_argument("--images-dir", default="data/test_images")
    ap.add_argument("--out", default="runs/test_images_out")
    ap.add_argument("--device", default=None)
    ap.add_argument("--independent", action="store_true",
                    help="reset temporal state per image (unrelated stills)")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="frame rate of the sequence (sets the model's dt "
                         "input; deployment loop runs 90-120)")
    ap.add_argument("--vis-threshold", type=float, default=0.5)
    args = ap.parse_args()

    cfg = load_config(args.config)
    spec = load_config(args.gate_spec)
    device = pick_device(args.device)
    d = cfg["data"]
    mh, mw = int(d.get("height", 192)), int(d.get("width", 320))

    paths = sorted(p for p in Path(args.images_dir).iterdir()
                   if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise SystemExit(
            f"no images found in {args.images_dir} — drop .png/.jpg files "
            f"there first (see data/test_images/README.md)")

    model = build_gateposenet(cfg["model"]).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{len(paths)} images | model {mw}x{mh} | device {device} | "
          f"gate outer {spec['gate']['outer_m']} m")

    h = None
    # ego = [v, omega, dt]: motion unknown for arbitrary images (zeros, the
    # model is trained with ego dropout) but the frame interval IS known.
    ego = torch.zeros(1, 7, device=device)
    ego[0, 6] = 1.0 / float(getattr(args, "fps", 30.0) or 30.0)
    records = []
    with torch.no_grad():
        for p in paths:
            raw = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if raw is None:
                print(f"  skip unreadable {p.name}")
                continue
            H0, W0 = raw.shape[:2]
            rgb = cv2.cvtColor(cv2.resize(raw, (mw, mh),
                                          interpolation=cv2.INTER_AREA),
                               cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(rgb.transpose(2, 0, 1))[None].float() / 255.0
            if args.independent:
                h = None
            out, h = model.step(x.to(device), ego, h)

            vis_p = float(torch.sigmoid(out["visible_logit"])[0])
            visible = vis_p >= args.vis_threshold
            scale = np.array([W0, H0], dtype=np.float64)
            corners = out["corners_uv"][0].cpu().numpy() * scale
            corner_in = torch.sigmoid(
                out["corner_inside_logit"])[0].cpu().numpy()
            center = out["center_uv"][0].cpu().numpy() * scale
            pos = out["position"][0].cpu().numpy()
            R = out["R"][0].cpu().numpy()
            axis = R[:, 2]                     # fly-through axis (away)
            rec = {
                "file": p.name,
                "visible_prob": vis_p,
                "visible": bool(visible),
                "visible_frac_pred": float(out["visible_frac"][0]),
                "corners_px": corners.tolist(),
                "corner_inside_prob": corner_in.tolist(),
                "center_px": center.tolist(),
                "position_cam_m": pos.tolist(),
                "distance_m": float(np.linalg.norm(pos)),
                "R_cam_gate": R.tolist(),
                "flythrough_axis_cam": axis.tolist(),
                "k_scale_pred": out["k_scale"][0].cpu().numpy().tolist(),
            }
            records.append(rec)

            # annotate (colors: visible green, low-confidence orange)
            im = raw.copy()
            col = (0, 220, 0) if visible else (0, 140, 255)
            pts = corners.astype(int)
            cv2.polylines(im, [pts.reshape(-1, 1, 2)], True, col, 2)
            for uv, ip in zip(pts, corner_in):
                cv2.circle(im, tuple(uv), 4, col, -1 if ip > 0.5 else 1)
            cv2.drawMarker(im, tuple(center.astype(int)), (255, 0, 255),
                           cv2.MARKER_CROSS, 16, 2)
            cv2.rectangle(im, (0, 0), (W0, 22), (0, 0, 0), -1)
            cv2.putText(
                im,
                f"vis={vis_p:.2f}  d={rec['distance_m']:.2f}m  "
                f"pos=[{pos[0]:+.2f} {pos[1]:+.2f} {pos[2]:+.2f}]m  "
                f"axis_z={axis[2]:+.2f}",
                (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(out_dir / p.name), im)

    with open(out_dir / "predictions.json", "w") as f:
        json.dump({
            "checkpoint": str(args.checkpoint),
            "gate_spec": spec["gate"],
            "note": ("position/R are CAMERA-OPTICAL frame; convert to body "
                     "with the actual mount tilt "
                     "(sam3 synthetic.pose_gt.optical_to_body). Ego-motion "
                     "input was zeroed."),
            "frames": records,
        }, f, indent=2)
    n_vis = sum(1 for r in records if r["visible"])
    print(f"done: {len(records)} frames ({n_vis} visible) -> {out_dir}/ "
          f"(+ predictions.json)")


if __name__ == "__main__":
    main()
