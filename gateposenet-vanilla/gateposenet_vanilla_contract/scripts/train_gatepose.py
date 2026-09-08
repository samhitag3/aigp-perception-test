#!/usr/bin/env python3
"""Train GatePoseNet (temporal multimodal gate-pose regression).

Usage:
    python scripts/train_gatepose.py --config configs/gateposenet_traj.yaml
    python scripts/train_gatepose.py --config ... --epochs 5 --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatenet.schedule import build_scheduler
from gatenet.utils import load_checkpoint, load_config, pick_device, set_seed
from gateposenet.engine import build_datasets, fit, make_loaders, make_optimizer
from gateposenet.data_sources import build_canonical_dataset, canonical_source_specs
from gateposenet.contract_reporting import evaluate_gateposenet_contract
from gateposenet.losses import GatePoseLoss
from gateposenet.model_single import \
    build_gateposenet_single as build_gateposenet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--dataset-root", default=None,
                    help="override data.dataset_root for single-source canonical datasets")
    ap.add_argument("--source-root", action="append", default=[], metavar="NAME=PATH",
                    help="override a named data.sources root; may be repeated")
    ap.add_argument("--train-manifest", default=None)
    ap.add_argument("--val-manifest", default=None)
    ap.add_argument("--test-manifest", default=None)
    ap.add_argument("--run-dir", default=None,
                    help="override output.dir")
    ap.add_argument("--skip-final-eval", action="store_true",
                    help="do not write evaluation.json/per_sample_metrics.csv")
    ap.add_argument("--final-eval-split", choices=["validation", "test"],
                    default="validation",
                    help="standard report split; keep validation during model selection")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.dataset_root is not None:
        cfg.setdefault("data", {})["dataset_root"] = args.dataset_root
    if args.source_root:
        sources = cfg.setdefault("data", {}).get("sources") or []
        by_name = {str(x.get("name")): x for x in sources if isinstance(x, dict)}
        for item in args.source_root:
            if "=" not in item:
                raise ValueError("--source-root must be NAME=PATH")
            name, path = item.split("=", 1)
            if name not in by_name:
                raise KeyError(f"unknown configured source {name!r}; choices={sorted(by_name)}")
            by_name[name]["root"] = path
    if args.train_manifest is not None:
        cfg["data"]["train_manifest"] = args.train_manifest
        for src in cfg["data"].get("sources", []):
            src["train_manifest"] = args.train_manifest
    if args.val_manifest is not None:
        cfg["data"]["val_manifest"] = args.val_manifest
        for src in cfg["data"].get("sources", []):
            src["val_manifest"] = args.val_manifest
    if args.test_manifest is not None:
        cfg["data"]["test_manifest"] = args.test_manifest
        for src in cfg["data"].get("sources", []):
            src["test_manifest"] = args.test_manifest
    if args.run_dir is not None:
        cfg.setdefault("output", {})["dir"] = args.run_dir
    set_seed(cfg["train"].get("seed", 42))
    device = pick_device(args.device)
    epochs = args.epochs or cfg["train"]["epochs"]
    out_dir = Path(cfg.get("output", {}).get("dir", "runs/gateposenet"))

    train_ds, val_ds = build_datasets(cfg)
    print(f"windows: train={len(train_ds)} val={len(val_ds)} "
          f"(window={cfg['data'].get('window', 8)})")
    train_loader, val_loader = make_loaders(cfg, train_ds, val_ds)

    model = build_gateposenet(cfg["model"]).to(device)
    print(f"GatePoseNet parameters: {model.num_parameters()/1e6:.3f} M "
          f"| device: {device}")
    criterion = GatePoseLoss(**cfg.get("loss", {})).to(device)
    optimizer = make_optimizer(cfg, model)
    scheduler = build_scheduler(optimizer, cfg.get("schedule"))

    start_epoch, best = 0, float("inf")
    if args.resume:
        ck = load_checkpoint(args.resume, model, optimizer, scheduler,
                             map_location=device)
        start_epoch = ck.get("epoch", 0) + 1
        best = ck.get("best_metric", float("inf"))
        print(f"resumed from {args.resume} @ epoch {start_epoch}")

    best = fit(cfg, model, train_loader, val_loader, criterion, optimizer,
               scheduler, device, out_dir, epochs, start_epoch, best)
    print(f"done. best val pos_err_rel={best:.4f}  -> {out_dir}/best.pt")

    # Canonical experiment contract: every completed canonical-data training
    # run writes evaluation.json + per_sample_metrics.csv using best.pt.
    if not args.skip_final_eval and canonical_source_specs(cfg.get("data", {})):
        d = cfg["data"]
        split_name = args.final_eval_split
        eval_ds = build_canonical_dataset(
            d, split_name,
            stride=int(d.get("test_stride", d.get("window", 8))),
            ego_dropout=0.0)
        nw = int(cfg["train"].get("num_workers", 4))
        eval_loader = __import__("torch").utils.data.DataLoader(
            eval_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False,
            num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
        load_checkpoint(out_dir / "best.pt", model, map_location=device)
        evaluate_gateposenet_contract(
            model, eval_loader, device, cfg, split_name, out_dir,
            checkpoint_path=out_dir / "best.pt")
        print(f"standard report -> {out_dir}/evaluation.json")
        print(f"per-sample metrics -> {out_dir}/per_sample_metrics.csv")


if __name__ == "__main__":
    main()
