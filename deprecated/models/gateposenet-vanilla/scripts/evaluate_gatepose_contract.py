#!/usr/bin/env python3
"""Evaluate vanilla GatePoseNetSingle on a canonical sequence split.

Writes the project-standard:
  evaluation.json
  per_sample_metrics.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatenet.utils import load_checkpoint, load_config, pick_device
from gateposenet.data_sources import build_canonical_dataset
from gateposenet.contract_reporting import evaluate_gateposenet_contract
from gateposenet.model_single import build_gateposenet_single


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gateposenet_vanilla_contract.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--split", choices=["train", "validation", "test"], default="test")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.dataset_root:
        cfg["data"]["dataset_root"] = args.dataset_root
    d = cfg["data"]
    if args.manifest is not None:
        key = {"train": "train_manifest", "validation": "val_manifest", "test": "test_manifest"}[args.split]
        d[key] = args.manifest
        for src in d.get("sources", []):
            src[key] = args.manifest

    stride = int(d.get("test_stride", d.get("window", 8)))
    ds = build_canonical_dataset(d, args.split, stride=stride, ego_dropout=0.0)
    bs = args.batch_size or int(cfg.get("train", {}).get("batch_size", 16))
    nw = args.num_workers if args.num_workers is not None else int(cfg.get("train", {}).get("num_workers", 4))
    loader = torch.utils.data.DataLoader(
        ds, batch_size=bs, shuffle=False, num_workers=nw,
        pin_memory=True, persistent_workers=nw > 0)

    device = pick_device(args.device)
    model = build_gateposenet_single(cfg["model"]).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)

    out = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).resolve().parent
    report = evaluate_gateposenet_contract(
        model, loader, device, cfg, args.split, out,
        checkpoint_path=args.checkpoint)
    print(f"wrote {out / 'evaluation.json'}")
    print(f"wrote {out / 'per_sample_metrics.csv'}")
    p = report["evaluation"]["pose"]["translation"]
    print(f"translation median={p['median_error_m']} m mean={p['mean_error_m']} m")


if __name__ == "__main__":
    main()
