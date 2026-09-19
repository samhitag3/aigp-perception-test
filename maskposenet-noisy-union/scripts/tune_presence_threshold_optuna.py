#!/usr/bin/env python3
"""Post-training Optuna calibration for MaskPoseNet-MG presence threshold.

This version is geometry-aware: in addition to normal presence precision/recall,
it compares each Hungarian-matched prediction's 4 outer gate corners against the
GT corners (D4/square-symmetry aware).  The default objective strongly prioritizes
keypoint-correct detections, so high-confidence predictions with wildly wrong gate
quadrilaterals push the selected presence threshold upward when confidence allows it.

The network is run over the calibration split once.  Everything needed by the
threshold sweep is then cached in RAM; Optuna only searches the scalar presence
threshold and does not retrain the model.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatenet.utils import load_checkpoint, load_config, pick_device
from gateposenet.losses import CORNER_PERMS
from gateposenet.losses_mg import GatePoseMGLoss
from gateposenet.mg_engine_contract import build_mg_dataset, set_manifest
from gateposenet.model import build_gateposenet_mg
from gateposenet.geometry_filter import GeometryFilterConfig, config_as_dict


def metrics_at_threshold(cache: dict[str, np.ndarray], th: float) -> dict[str, float]:
    prob = cache["prob"]
    target_q = cache["target_q"]
    visible_q = cache["visible_q"]
    far_q = cache["far_q"]
    valid_frame = cache["valid_frame"]

    # Geometry-aware cached labels/measurements.
    # kp_eval_q: matched + visible + all 4 GT corners geometrically valid.
    # kp_good_q: kp_eval_q and the predicted 4-corner geometry is close enough.
    kp_eval_q = cache["kp_eval_q"]
    kp_good_q = cache["kp_good_q"]
    corner_err_px_q = cache["corner_err_px_q"]
    corner_err_frac_q = cache["corner_err_frac_q"]

    pred = prob >= th
    vf = valid_frame[..., None]

    # ------------------------------------------------------------------
    # Original presence/detection metrics.
    # ------------------------------------------------------------------
    tp = int(np.logical_and(pred, target_q & vf).sum())
    fp = int(np.logical_and(pred, (~target_q) & vf).sum())
    fn = int(np.logical_and(~pred, target_q & vf).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    beta2 = 4.0
    f2 = (1.0 + beta2) * precision * recall / max(
        beta2 * precision + recall, 1e-12
    )

    visible_total = int((visible_q & vf).sum())
    visible_hits = int((pred & visible_q & vf).sum())
    visible_recall = visible_hits / max(visible_total, 1)

    far_total = int((far_q & vf).sum())
    far_hits = int((pred & far_q & vf).sum())
    far_recall = far_hits / max(far_total, 1)

    n_pred = pred.sum(axis=-1)
    n_gt = target_q.sum(axis=-1)
    count_acc = (
        float((n_pred[valid_frame] == n_gt[valid_frame]).mean())
        if valid_frame.any()
        else 0.0
    )

    far_balanced = 0.50 * f1 + 0.35 * far_recall + 0.15 * count_acc

    # ------------------------------------------------------------------
    # Keypoint/geometry-aware metrics.
    #
    # A GT gate only becomes a geometry TP when:
    #   1) Hungarian matched it to this query,
    #   2) it is visible and has 4 valid GT corners,
    #   3) the presence threshold accepts it, and
    #   4) predicted corners pass BOTH absolute-pixel and scale-normalized
    #      error limits.
    #
    # Active unmatched queries and active matched-but-bad-geometry queries
    # are keypoint false positives.  This directly punishes the giant bogus
    # quadrilaterals that are distracting in inference overlays.
    # ------------------------------------------------------------------
    kp_eval = kp_eval_q & vf
    kp_good = kp_good_q & vf
    kp_bad = kp_eval & (~kp_good_q)

    kp_tp = int((pred & kp_good).sum())
    bad_geometry_active = int((pred & kp_bad).sum())
    unmatched_active = int((pred & (~target_q) & vf).sum())

    # Every evaluable GT gate that is not an accepted good-geometry detection
    # is a keypoint FN (including an accepted query whose corners are wrong).
    kp_total = int(kp_eval.sum())
    kp_fn = max(kp_total - kp_tp, 0)
    kp_fp = bad_geometry_active + unmatched_active

    kp_precision = kp_tp / max(kp_tp + kp_fp, 1)
    kp_recall = kp_tp / max(kp_total, 1)
    kp_f1 = 2.0 * kp_precision * kp_recall / max(
        kp_precision + kp_recall, 1e-12
    )

    active_eval = pred & kp_eval
    n_active_eval = int(active_eval.sum())
    bad_geometry_active_rate = bad_geometry_active / max(n_active_eval, 1)

    active_corner_err_px = corner_err_px_q[active_eval]
    active_corner_err_frac = corner_err_frac_q[active_eval]
    if active_corner_err_px.size:
        mean_active_corner_err_px = float(np.nanmean(active_corner_err_px))
        p95_active_corner_err_px = float(np.nanpercentile(active_corner_err_px, 95))
        mean_active_corner_err_frac = float(np.nanmean(active_corner_err_frac))
        p95_active_corner_err_frac = float(np.nanpercentile(active_corner_err_frac, 95))
    else:
        mean_active_corner_err_px = float("nan")
        p95_active_corner_err_px = float("nan")
        mean_active_corner_err_frac = float("nan")
        p95_active_corner_err_frac = float("nan")

    # Keypoints dominate the objective.  Ordinary detection F1, far recall,
    # and count accuracy remain as guard rails so the tuner cannot get a good
    # score simply by suppressing nearly everything.
    keypoint_balanced = (
        0.70 * kp_f1
        + 0.15 * far_recall
        + 0.10 * f1
        + 0.05 * count_acc
    )

    return {
        "threshold": float(th),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f2": float(f2),
        "visible_recall": float(visible_recall),
        "far_recall": float(far_recall),
        "count_acc": float(count_acc),
        "far_balanced": float(far_balanced),
        "keypoint_precision": float(kp_precision),
        "keypoint_recall": float(kp_recall),
        "keypoint_f1": float(kp_f1),
        "keypoint_balanced": float(keypoint_balanced),
        "bad_geometry_active_rate": float(bad_geometry_active_rate),
        "mean_active_corner_err_px": mean_active_corner_err_px,
        "p95_active_corner_err_px": p95_active_corner_err_px,
        "mean_active_corner_err_frac": mean_active_corner_err_frac,
        "p95_active_corner_err_frac": p95_active_corner_err_frac,
        "kp_evaluable_gt": kp_total,
        "kp_tp": kp_tp,
        "kp_fp": kp_fp,
        "kp_fn": kp_fn,
        "bad_geometry_active": bad_geometry_active,
        "unmatched_active": unmatched_active,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _scatter_bool(mask_gate: torch.Tensor, safe_assign: torch.Tensor, q: int) -> torch.Tensor:
    """Scatter per-GT boolean values to query slots without duplicate overwrite.

    Invalid GT slots have safe_assign==0, so bool scatter_ can accidentally
    overwrite a valid query-0 True with later False values.  scatter_add_ avoids
    that problem and then converts back to bool.
    """
    out = torch.zeros(
        *mask_gate.shape[:2], q, dtype=torch.int16, device=mask_gate.device
    )
    out.scatter_add_(2, safe_assign, mask_gate.to(torch.int16))
    return out > 0


@torch.no_grad()
def collect_cache(
    model,
    loader,
    criterion,
    device,
    img_wh: tuple[int, int],
    max_corner_err_px: float,
    max_corner_err_frac: float,
) -> dict[str, np.ndarray]:
    probs = []
    targets = []
    visibles = []
    fars = []
    valid_frames = []
    kp_evals = []
    kp_goods = []
    corner_err_pxs = []
    corner_err_fracs = []

    far_start = float(getattr(criterion, "far_start_m", 8.0))
    amp = device.type == "cuda"
    W, H = [float(v) for v in img_wh]
    scale = torch.tensor([W, H], dtype=torch.float32, device=device)
    perms = CORNER_PERMS.to(device)

    model.eval()
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        with torch.amp.autocast("cuda", enabled=amp):
            pred = model(batch["image"], batch["ego"])

        # Presence probabilities are safe to derive from the AMP forward pass,
        # but Hungarian matching uses torch.cdist(), which is not implemented
        # for CUDA Half on this environment.  Match in float32.
        prob = torch.sigmoid(pred["presence_logit"].float())
        pred_match = {
            k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
            for k, v in pred.items()
        }
        assign = criterion._match(pred_match, batch)

        B, T, Q = prob.shape
        N = assign.shape[2]
        safe = assign.clamp_min(0)
        frame_ok = batch["frame_ok"] > 0.5
        matched = (
            (assign >= 0)
            & (batch["g_valid"] > 0.5)
            & frame_ok[..., None]
        )
        visible = matched & (batch["g_visible"] > 0.5)
        ranges = batch["g_position"].float().norm(dim=-1)
        far = visible & (ranges >= far_start)

        target_q = _scatter_bool(matched, safe, Q)
        visible_q = _scatter_bool(visible, safe, Q)
        far_q = _scatter_bool(far, safe, Q)

        # --------------------------------------------------------------
        # Compare the predicted 4 outer corners for each Hungarian-matched
        # query against the GT gate's 4 corners.
        #
        # Use minimum mean corner error over all D4 square permutations so
        # a correct square is not penalized merely because its corner labels
        # are rotated/flipped by a symmetry-equivalent representation.
        # --------------------------------------------------------------
        pred_corners = pred["corners_uv"].float()              # B,T,Q,4,2
        gt_corners = batch["g_corners"].float()                # B,T,N,4,2

        idx = safe[..., None, None].expand(B, T, N, 4, 2)
        matched_pred_corners = pred_corners.gather(2, idx)      # B,T,N,4,2

        gt_perm = gt_corners[:, :, :, perms, :]                 # B,T,N,8,4,2
        diff_px = (
            matched_pred_corners[:, :, :, None, :, :] - gt_perm
        ) * scale.view(1, 1, 1, 1, 1, 2)
        err_by_sym_px = diff_px.norm(dim=-1).mean(dim=-1)       # B,T,N,8
        corner_err_px, _ = err_by_sym_px.min(dim=-1)            # B,T,N

        # Normalize by the GT gate's mean diagonal length.  This prevents a
        # fixed pixel tolerance from being too loose for small/far gates and
        # too strict for very large/near gates.
        gt_px = gt_corners * scale.view(1, 1, 1, 2)
        diag_02 = (gt_px[..., 0, :] - gt_px[..., 2, :]).norm(dim=-1)
        diag_13 = (gt_px[..., 1, :] - gt_px[..., 3, :]).norm(dim=-1)
        gt_diag_px = 0.5 * (diag_02 + diag_13)
        corner_err_frac = corner_err_px / gt_diag_px.clamp_min(1.0)

        all_four_gt_corners_ok = (batch["g_corner_ok"] > 0.5).all(dim=-1)
        kp_eval_gate = visible & all_four_gt_corners_ok
        kp_good_gate = (
            kp_eval_gate
            & (corner_err_px <= max_corner_err_px)
            & (corner_err_frac <= max_corner_err_frac)
        )

        kp_eval_q = _scatter_bool(kp_eval_gate, safe, Q)
        kp_good_q = _scatter_bool(kp_good_gate, safe, Q)

        # Store the actual error on the matched query slot; NaN elsewhere.
        corner_err_px_q = torch.full(
            (B, T, Q), float("nan"), dtype=torch.float32, device=device
        )
        corner_err_frac_q = torch.full_like(corner_err_px_q, float("nan"))
        nz = torch.nonzero(kp_eval_gate, as_tuple=False)
        if nz.numel() > 0:
            bi, ti, gi = nz.unbind(dim=1)
            qi = assign[bi, ti, gi]
            corner_err_px_q[bi, ti, qi] = corner_err_px[bi, ti, gi]
            corner_err_frac_q[bi, ti, qi] = corner_err_frac[bi, ti, gi]

        probs.append(prob.cpu().numpy())
        targets.append(target_q.cpu().numpy())
        visibles.append(visible_q.cpu().numpy())
        fars.append(far_q.cpu().numpy())
        valid_frames.append(frame_ok.cpu().numpy())
        kp_evals.append(kp_eval_q.cpu().numpy())
        kp_goods.append(kp_good_q.cpu().numpy())
        corner_err_pxs.append(corner_err_px_q.cpu().numpy())
        corner_err_fracs.append(corner_err_frac_q.cpu().numpy())

    return {
        "prob": np.concatenate(probs, axis=0),
        "target_q": np.concatenate(targets, axis=0),
        "visible_q": np.concatenate(visibles, axis=0),
        "far_q": np.concatenate(fars, axis=0),
        "valid_frame": np.concatenate(valid_frames, axis=0),
        "kp_eval_q": np.concatenate(kp_evals, axis=0),
        "kp_good_q": np.concatenate(kp_goods, axis=0),
        "corner_err_px_q": np.concatenate(corner_err_pxs, axis=0),
        "corner_err_frac_q": np.concatenate(corner_err_fracs, axis=0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", choices=["validation", "test"], default="validation")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--min-th", type=float, default=0.05)
    ap.add_argument("--max-th", type=float, default=0.95)
    ap.add_argument(
        "--objective",
        choices=[
            "f1",
            "f2",
            "far_recall",
            "count_acc",
            "far_balanced",
            "keypoint_f1",
            "keypoint_balanced",
        ],
        default="keypoint_balanced",
    )
    ap.add_argument(
        "--max-corner-err-px",
        type=float,
        default=50.0,
        help="Mean 4-corner error must be <= this many native-image pixels.",
    )
    ap.add_argument(
        "--max-corner-err-frac",
        type=float,
        default=0.30,
        help=(
            "Mean 4-corner error divided by GT gate diagonal must be <= this. "
            "0.30 means 30%% of the GT gate diagonal."
        ),
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not 0.0 <= args.min_th < args.max_th <= 1.0:
        raise ValueError("require 0 <= min-th < max-th <= 1")
    if args.max_corner_err_px <= 0:
        raise ValueError("--max-corner-err-px must be > 0")
    if args.max_corner_err_frac <= 0:
        raise ValueError("--max-corner-err-frac must be > 0")

    cfg = load_config(args.config)
    if args.manifest:
        set_manifest(cfg, args.split, args.manifest)
    device = pick_device(args.device)

    ds = build_mg_dataset(cfg, args.split, ego_dropout=0.0)
    nw = int(cfg["train"].get("num_workers", 8))
    kw = dict(num_workers=nw, pin_memory=True)
    if nw > 0:
        kw["persistent_workers"] = True
    loader = DataLoader(
        ds,
        batch_size=int(cfg["train"].get("batch_size", 12)),
        shuffle=False,
        drop_last=False,
        **kw,
    )

    model = build_gateposenet_mg(cfg["model"]).to(device)
    ckpt = load_checkpoint(args.checkpoint, model, map_location=device)
    criterion = GatePoseMGLoss(**cfg.get("loss", {})).to(device)

    img_wh = (
        int(cfg["data"].get("native_width", 640)),
        int(cfg["data"].get("native_height", 360)),
    )

    print(
        f"caching presence + keypoint outputs: split={args.split} "
        f"windows={len(ds)} checkpoint_epoch={ckpt.get('epoch', -1)}"
    )
    print(
        f"keypoint quality limits: mean_corner_err <= {args.max_corner_err_px:.1f}px "
        f"AND <= {args.max_corner_err_frac:.3f} * GT diagonal"
    )

    cache = collect_cache(
        model,
        loader,
        criterion,
        device,
        img_wh=img_wh,
        max_corner_err_px=float(args.max_corner_err_px),
        max_corner_err_frac=float(args.max_corner_err_frac),
    )
    print(
        f"cached windows={cache['prob'].shape[0]} "
        f"T={cache['prob'].shape[1]} Q={cache['prob'].shape[2]}"
    )
    print(
        f"keypoint-evaluable matched gates={int(cache['kp_eval_q'].sum())} "
        f"geometry-good={int(cache['kp_good_q'].sum())}"
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name="maskposenet_presence_threshold_keypoints",
        storage=f"sqlite:///{(out / 'presence_threshold_study.db').resolve()}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        load_if_exists=True,
    )

    def objective(trial):
        th = trial.suggest_float("presence_threshold", args.min_th, args.max_th)
        m = metrics_at_threshold(cache, th)
        for k, v in m.items():
            if k != "threshold":
                # SQLite/Optuna user attrs should not receive NaN if avoidable.
                if isinstance(v, float) and not np.isfinite(v):
                    continue
                trial.set_user_attr(k, v)
        return float(m[args.objective])

    study.optimize(objective, n_trials=args.trials)
    best_th = float(study.best_params["presence_threshold"])
    best_metrics = metrics_at_threshold(cache, best_th)

    best_cfg = copy.deepcopy(cfg)
    best_cfg.setdefault("inference", {})["presence_threshold"] = best_th
    best_cfg["inference"].setdefault("mask_threshold", 0.5)
    # Presence calibration cannot reject a bad quad whose confidence is already
    # near 1.0. Preserve/add the runtime mask/keypoint consistency filter in the
    # generated deployment config so inference can remove those outputs.
    best_cfg["inference"].setdefault(
        "geometry_filter", config_as_dict(GeometryFilterConfig())
    )
    cfg_path = out / "best_config_with_threshold.yaml"
    cfg_path.write_text(
        yaml.safe_dump(best_cfg, sort_keys=False), encoding="utf-8"
    )

    payload = {
        "schema_version": "1.1.0-keypoint-aware",
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "split": args.split,
        "objective": args.objective,
        "best_value": float(study.best_value),
        "best_threshold": best_th,
        "keypoint_quality": {
            "comparison": "D4-symmetry-aware mean outer-corner error",
            "native_image_wh": [img_wh[0], img_wh[1]],
            "max_corner_err_px": float(args.max_corner_err_px),
            "max_corner_err_frac_of_gt_diagonal": float(args.max_corner_err_frac),
            "requires_visible_gate": True,
            "requires_all_4_valid_gt_corners": True,
        },
        "objective_weights": {
            "keypoint_f1": 0.70,
            "far_recall": 0.15,
            "presence_f1": 0.10,
            "count_acc": 0.05,
        },
        "metrics": best_metrics,
        "trials": len(study.trials),
    }
    (out / "best_presence_threshold.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )

    rows = []
    for t in study.trials:
        r = {"trial": t.number, "state": t.state.name, "value": t.value}
        r.update(t.params)
        r.update(t.user_attrs)
        rows.append(r)
    fields = sorted({k for r in rows for k in r}) if rows else ["trial", "value"]
    with (out / "presence_threshold_trials.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(json.dumps(payload, indent=2, allow_nan=True))
    print(f"best config -> {cfg_path}")


if __name__ == "__main__":
    main()
