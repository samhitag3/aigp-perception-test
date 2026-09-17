#!/usr/bin/env python3
"""Post-training Optuna calibration for the MaskPoseNet-MG presence threshold.

The model is run over the calibration split once. Presence probabilities and
Hungarian GT/query assignments are cached in RAM. Optuna then searches only the
scalar decision threshold, so trials are extremely cheap and do not retrain or
rerun the network.
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
from gateposenet.losses_mg import GatePoseMGLoss
from gateposenet.mg_engine_contract import build_mg_dataset, set_manifest
from gateposenet.model import build_gateposenet_mg


def metrics_at_threshold(cache: dict[str, np.ndarray], th: float) -> dict[str, float]:
    prob = cache["prob"]
    target_q = cache["target_q"]
    visible_q = cache["visible_q"]
    far_q = cache["far_q"]
    valid_frame = cache["valid_frame"]

    pred = prob >= th
    vf = valid_frame[..., None]

    tp = int(np.logical_and(pred, target_q & vf).sum())
    fp = int(np.logical_and(pred, (~target_q) & vf).sum())
    fn = int(np.logical_and(~pred, target_q & vf).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    beta2 = 4.0
    f2 = (1.0 + beta2) * precision * recall / max(beta2 * precision + recall, 1e-12)

    visible_total = int((visible_q & vf).sum())
    visible_hits = int((pred & visible_q & vf).sum())
    visible_recall = visible_hits / max(visible_total, 1)

    far_total = int((far_q & vf).sum())
    far_hits = int((pred & far_q & vf).sum())
    far_recall = far_hits / max(far_total, 1)

    n_pred = pred.sum(axis=-1)
    n_gt = target_q.sum(axis=-1)
    count_acc = float((n_pred[valid_frame] == n_gt[valid_frame]).mean()) if valid_frame.any() else 0.0

    # Competition-oriented threshold score: false positives still matter via F1,
    # while far-gate recall gets explicit weight because missing small/far gates
    # was the original failure mode. Count accuracy discourages trivially low thresholds.
    far_balanced = 0.50 * f1 + 0.35 * far_recall + 0.15 * count_acc

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
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


@torch.no_grad()
def collect_cache(model, loader, criterion, device) -> dict[str, np.ndarray]:
    probs, targets, visibles, fars, valid_frames = [], [], [], [], []
    far_start = float(getattr(criterion, "far_start_m", 8.0))
    amp = device.type == "cuda"

    model.eval()
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast("cuda", enabled=amp):
            pred = model(batch["image"], batch["ego"])

        prob = torch.sigmoid(pred["presence_logit"])
        assign = criterion._match(pred, batch)
        B, T, Q = prob.shape
        safe = assign.clamp_min(0)
        matched = (assign >= 0) & (batch["g_valid"] > 0.5) & (batch["frame_ok"][..., None] > 0.5)
        visible = matched & (batch["g_visible"] > 0.5)
        ranges = batch["g_position"].norm(dim=-1)
        far = visible & (ranges >= far_start)

        target_q = torch.zeros(B, T, Q, dtype=torch.bool, device=device)
        visible_q = torch.zeros_like(target_q)
        far_q = torch.zeros_like(target_q)
        target_q.scatter_(2, safe, matched)
        visible_q.scatter_(2, safe, visible)
        far_q.scatter_(2, safe, far)

        probs.append(prob.float().cpu().numpy())
        targets.append(target_q.cpu().numpy())
        visibles.append(visible_q.cpu().numpy())
        fars.append(far_q.cpu().numpy())
        valid_frames.append((batch["frame_ok"] > 0.5).cpu().numpy())

    return {
        "prob": np.concatenate(probs, axis=0),
        "target_q": np.concatenate(targets, axis=0),
        "visible_q": np.concatenate(visibles, axis=0),
        "far_q": np.concatenate(fars, axis=0),
        "valid_frame": np.concatenate(valid_frames, axis=0),
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
    ap.add_argument("--objective", choices=["f1", "f2", "far_recall", "count_acc", "far_balanced"], default="far_balanced")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not 0.0 <= args.min_th < args.max_th <= 1.0:
        raise ValueError("require 0 <= min-th < max-th <= 1")

    cfg = load_config(args.config)
    if args.manifest:
        set_manifest(cfg, args.split, args.manifest)
    device = pick_device(args.device)
    ds = build_mg_dataset(cfg, args.split, ego_dropout=0.0)
    nw = int(cfg["train"].get("num_workers", 8))
    kw = dict(num_workers=nw, pin_memory=True)
    if nw > 0:
        kw["persistent_workers"] = True
    loader = DataLoader(ds, batch_size=int(cfg["train"].get("batch_size", 12)), shuffle=False, drop_last=False, **kw)

    model = build_gateposenet_mg(cfg["model"]).to(device)
    ckpt = load_checkpoint(args.checkpoint, model, map_location=device)
    criterion = GatePoseMGLoss(**cfg.get("loss", {})).to(device)

    print(f"caching presence outputs: split={args.split} windows={len(ds)} checkpoint_epoch={ckpt.get('epoch', -1)}")
    cache = collect_cache(model, loader, criterion, device)
    print(f"cached windows={cache['prob'].shape[0]} T={cache['prob'].shape[1]} Q={cache['prob'].shape[2]}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name="maskposenet_presence_threshold",
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
                trial.set_user_attr(k, v)
        return float(m[args.objective])

    study.optimize(objective, n_trials=args.trials)
    best_th = float(study.best_params["presence_threshold"])
    best_metrics = metrics_at_threshold(cache, best_th)

    best_cfg = copy.deepcopy(cfg)
    best_cfg.setdefault("inference", {})["presence_threshold"] = best_th
    best_cfg["inference"].setdefault("mask_threshold", 0.5)
    cfg_path = out / "best_config_with_threshold.yaml"
    cfg_path.write_text(yaml.safe_dump(best_cfg, sort_keys=False), encoding="utf-8")

    payload = {
        "schema_version": "1.0.0",
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "split": args.split,
        "objective": args.objective,
        "best_value": float(study.best_value),
        "best_threshold": best_th,
        "metrics": best_metrics,
        "trials": len(study.trials),
    }
    (out / "best_presence_threshold.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    rows = []
    for t in study.trials:
        r = {"trial": t.number, "state": t.state.name, "value": t.value}
        r.update(t.params)
        r.update(t.user_attrs)
        rows.append(r)
    fields = sorted({k for r in rows for k in r}) if rows else ["trial", "value"]
    with (out / "presence_threshold_trials.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(json.dumps(payload, indent=2))
    print(f"best config -> {cfg_path}")


if __name__ == "__main__":
    main()
