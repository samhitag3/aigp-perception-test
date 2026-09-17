#!/usr/bin/env python3
"""Train unchanged GatePoseNet-MG directly on canonical contract datasets."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatenet.utils import load_config, pick_device, set_seed
from gateposenet.losses_mg import GatePoseMGLoss
from gateposenet.mg_engine_contract import (fit_mg, make_loaders, make_optimizer,
                                            make_scheduler, set_manifest)
from gateposenet.model import build_gateposenet_mg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--train-manifest", default=None)
    ap.add_argument("--val-manifest", default=None)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.train_manifest:
        set_manifest(cfg, "train", args.train_manifest)
    if args.val_manifest:
        set_manifest(cfg, "validation", args.val_manifest)
    cfg.setdefault("output", {})["dir"] = args.run_dir
    seed = int(cfg["train"].get("seed", 42)); set_seed(seed)
    device = pick_device(args.device)

    max_gates = int(cfg["data"].get("max_gates", cfg["model"].get("n_queries", 8)))
    n_queries = int(cfg["model"].get("n_queries", 8))
    if max_gates > n_queries:
        raise ValueError(f"max_gates={max_gates} > n_queries={n_queries}")

    train_ds, val_ds, tl, vl = make_loaders(cfg)
    print(f"device={device} train_windows={len(train_ds)} val_windows={len(val_ds)} "
          f"max_gates={max_gates} queries={n_queries}")
    model = build_gateposenet_mg(cfg["model"]).to(device)
    print(f"GatePoseNet-MG parameters: {model.num_parameters()/1e6:.3f} M")
    criterion = GatePoseMGLoss(**cfg.get("loss", {})).to(device)
    optimizer = make_optimizer(cfg, model)
    scheduler = make_scheduler(cfg, optimizer)
    out = Path(args.run_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    epochs = int(args.epochs or cfg["train"].get("epochs", 60))
    best = fit_mg(cfg, model, tl, vl, criterion, optimizer, scheduler,
                  device, out, epochs, tag="mg")
    sel_name = cfg.get("selection", {}).get("metric", "target_pos_err_m")
    print(f"done. best {sel_name}={best:.4f} -> {out/'best.pt'}")


if __name__ == "__main__":
    main()
