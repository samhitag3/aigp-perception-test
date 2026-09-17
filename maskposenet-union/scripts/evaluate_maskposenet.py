#!/usr/bin/env python3
"""Evaluate MaskPoseNet-MG on a canonical validation/test split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatenet.utils import load_checkpoint, load_config, pick_device
from gateposenet.losses_mg import GatePoseMGLoss
from gateposenet.mg_engine_contract import build_mg_dataset, set_manifest, validate_mg
from gateposenet.model import build_gateposenet_mg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", choices=["validation", "test"], default="validation")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--presence-th", type=float, default=None,
                    help="Presence decision threshold; defaults to config inference.presence_threshold or 0.5")
    ap.add_argument("--mask-th", type=float, default=None,
                    help="Mask binarization threshold; defaults to config inference.mask_threshold or 0.5")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.manifest:
        set_manifest(cfg, args.split, args.manifest)
    device = pick_device(args.device)
    ds = build_mg_dataset(cfg, args.split, ego_dropout=0.0)
    nw = int(cfg["train"].get("num_workers", 8))
    kw = dict(num_workers=nw, pin_memory=True)
    if nw > 0:
        kw["persistent_workers"] = True
    loader = DataLoader(ds, batch_size=int(cfg["train"].get("batch_size", 12)),
                        shuffle=False, drop_last=False, **kw)
    model = build_gateposenet_mg(cfg["model"]).to(device)
    ckpt = load_checkpoint(args.checkpoint, model, map_location=device)
    criterion = GatePoseMGLoss(**cfg.get("loss", {})).to(device)
    infer_cfg = cfg.get("inference", {})
    presence_th = float(args.presence_th if args.presence_th is not None else infer_cfg.get("presence_threshold", 0.5))
    mask_th = float(args.mask_th if args.mask_th is not None else infer_cfg.get("mask_threshold", 0.5))
    metrics = validate_mg(model, loader, criterion, device, presence_th=presence_th, mask_th=mask_th)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0.0",
        "model": "MaskPoseNet-MG",
        "split": args.split,
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "num_windows": len(ds),
        "presence_threshold": presence_th,
        "mask_threshold": mask_th,
        "metrics": metrics,
    }
    (out/"evaluation_mg.json").write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote -> {out/'evaluation_mg.json'}")


if __name__ == "__main__":
    main()
