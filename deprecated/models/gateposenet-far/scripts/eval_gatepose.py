#!/usr/bin/env python3
"""Evaluate a trained GatePoseNet checkpoint on a trajectory dataset.

Reports symmetry-aware pose metrics (visible + blind), range-bucketed relative
position error, and a comparison against the classical corners->PnP baseline
computed from the model's OWN predicted corners (the calibrated deployment
path), sanity-checked against PnP on the GT corners.

Usage:
    python scripts/eval_gatepose.py --config configs/gateposenet_traj.yaml \
        --checkpoint runs/gateposenet_traj/best.pt [--roots DIR ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatenet.utils import load_checkpoint, load_config, pick_device
from gateposenet.dataset import GateSequenceDataset
from gateposenet.engine import _to_device
from gateposenet.losses import GatePoseLoss
from gateposenet.metrics import merge_metric_acc, pose_metrics, reduce_metrics
from gateposenet.model_single import \
    build_gateposenet_single as build_gateposenet

AIGP_GATE_OUTER_M = 2.7


def pnp_position(corners_px: np.ndarray, K: np.ndarray,
                 gate_m: float = AIGP_GATE_OUTER_M):
    """Classical PnP position from 4 corner pixels (calibrated path)."""
    h = gate_m / 2.0
    # gate body frame: +X right, +Y down, +Z fly-through; TL,TR,BR,BL
    obj = np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]],
                   dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, corners_px.astype(np.float64), K, None,
                                  flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    return tvec.reshape(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--roots", nargs="*", default=None,
                    help="dataset roots (default: config val_roots)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = pick_device(args.device)
    d = cfg["data"]
    roots = args.roots or d["val_roots"]
    ds = GateSequenceDataset(
        roots, window=int(d.get("window", 8)), stride=int(d.get("window", 8)),
        size=(int(d.get("height", 192)), int(d.get("width", 320))),
        pose_sup_max_m=float(d.get("pose_sup_max_m", 20.0)),
        nominal_focal=float(d.get("nominal_focal", 320.0)))
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                         num_workers=4)

    model = build_gateposenet(cfg["model"]).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    criterion = GatePoseLoss(**cfg.get("loss", {})).to(device)

    W, H = int(d.get("native_width", 640)), int(d.get("native_height", 360))
    K_nom = np.array([[d.get("nominal_focal", 320.0), 0, W / 2.0],
                      [0, d.get("nominal_focal", 320.0), H / 2.0],
                      [0, 0, 1.0]])

    acc: dict = {}
    pnp_pred_err, pnp_pred_err_nom, pnp_gt_err = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            pred = model(batch["image"], batch["ego"])
            merge_metric_acc(acc, pose_metrics(pred, batch, (W, H)))

            # PnP path on the LAST frame of each window (deployment style).
            # Two K variants: the frame's TRUE (jittered) K isolates corner
            # quality; the NOMINAL K is the deployed-calibration case.
            vis = batch["visible"][:, -1] > 0.5
            for b in range(batch["image"].shape[0]):
                if not bool(vis[b]):
                    continue
                gt_pos = batch["position"][b, -1].cpu().numpy()
                scale = batch["img_wh"][b, -1].cpu().numpy().astype(np.float64)
                K_true = batch["K"][b, -1].cpu().numpy().astype(np.float64)
                c_pred = pred["corners_uv"][b, -1].cpu().numpy() * scale
                c_gt = batch["corners_uv"][b, -1].cpu().numpy() * scale
                p1 = pnp_position(c_pred, K_true)
                p1n = pnp_position(c_pred, K_nom)
                p2 = pnp_position(c_gt, K_true)
                if p1 is not None:
                    pnp_pred_err.append(float(np.linalg.norm(p1 - gt_pos)))
                if p1n is not None:
                    pnp_pred_err_nom.append(
                        float(np.linalg.norm(p1n - gt_pos)))
                if p2 is not None:
                    pnp_gt_err.append(float(np.linalg.norm(p2 - gt_pos)))

    m = reduce_metrics(acc)
    print("=" * 72)
    print(f"GatePoseNet eval on {roots}")
    print(f"  windows                    : {len(ds)}")
    print(f"  visible pos err            : mean {m.get('pos_err_m', float('nan')):.3f} m | "
          f"median {m.get('pos_err_m_med', float('nan')):.3f} m")
    print(f"  visible pos err (relative) : mean {m.get('pos_err_rel', float('nan')):.4f} | "
          f"median {m.get('pos_err_rel_med', float('nan')):.4f}")
    print(f"  BLIND pos err (memory)     : mean {m.get('pos_err_m_blind', float('nan')):.3f} m | "
          f"median {m.get('pos_err_m_blind_med', float('nan')):.3f} m")
    print(f"  rotation err (sym-aware)   : mean {m.get('rot_err_deg', float('nan')):.2f} deg | "
          f"median {m.get('rot_err_deg_med', float('nan')):.2f} deg")
    print(f"  fly-through axis err       : mean {m.get('normal_err_deg', float('nan')):.2f} deg | "
          f"median {m.get('normal_err_deg_med', float('nan')):.2f} deg")
    print(f"  center err                 : mean {m.get('center_err_px', float('nan')):.1f} px | "
          f"median {m.get('center_err_px_med', float('nan')):.1f} px")
    print(f"  corner err (sym-aware)     : mean {m.get('corner_err_px', float('nan')):.1f} px | "
          f"median {m.get('corner_err_px_med', float('nan')):.1f} px")
    print(f"  visibility accuracy        : {m.get('vis_acc', float('nan')):.3f}")
    for lo, hi in [(0, 3), (3, 6), (6, 10), (10, 25)]:
        k = f"pos_rel_{lo}-{hi}m"
        if k in m:
            print(f"  pos err rel @ {lo:>2}-{hi:<2} m      : "
                  f"mean {m[k]:.4f} | median {m[k + '_med']:.4f}")
    if pnp_pred_err:
        print(f"  PnP(pred corners, true K)  : mean {np.mean(pnp_pred_err):.3f} m | "
              f"median {np.median(pnp_pred_err):.3f} m  (n={len(pnp_pred_err)})")
    if pnp_pred_err_nom:
        print(f"  PnP(pred corners, nom. K)  : mean {np.mean(pnp_pred_err_nom):.3f} m | "
              f"median {np.median(pnp_pred_err_nom):.3f} m  (deployed calib)")
    if pnp_gt_err:
        print(f"  PnP(GT corners, true K)    : mean {np.mean(pnp_gt_err):.3f} m | "
              f"median {np.median(pnp_gt_err):.3f} m  (sanity, expect ~0)")
    print("=" * 72)


if __name__ == "__main__":
    main()
