from __future__ import annotations
import argparse
from pathlib import Path
import torch

from gateperception.models import build_segmentation_model
from gateperception.training.builders import build_seg_loaders, seg_step, seg_val
from gateperception.training.engine import train_loop
from gateperception.training.evaluate_models import evaluate_segmentation_model
import json
from gateperception.utils.config import load_yaml, save_yaml
from gateperception.utils.seed import seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--init-checkpoint", default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    _, _, train_loader, val_loader = build_seg_loaders(cfg)
    model = build_segmentation_model(cfg).to(device)
    if args.init_checkpoint:
        ck = torch.load(args.init_checkpoint, map_location="cpu")
        model.load_state_dict(ck["model"], strict=True)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"].get("weight_decay", 1e-4)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(cfg["training"]["epochs"])))
    run_dir = Path(args.run_dir)
    save_yaml(cfg, run_dir / "config.yaml")
    result = train_loop(model, train_loader, val_loader, opt, int(cfg["training"]["epochs"]), device, run_dir, cfg, seg_step, seg_val, amp=bool(cfg["training"].get("amp", True)), scheduler=sched)
    best = torch.load(run_dir / "best.pt", map_location="cpu"); model.load_state_dict(best["model"]); model.to(device)
    metrics = evaluate_segmentation_model(model, val_loader, device, cfg)
    report = {"schema_version":"1.0.0","model_name":"temporal_gate_instance_segmenter","evaluation_split":"validation","training":{"best_epoch":best["epoch"],"best_val_metric":best["best_metric"],"training_time_seconds":result["training_time_seconds"]},"evaluation":metrics}
    with open(run_dir / "evaluation.json", "w", encoding="utf-8") as f: json.dump(report, f, indent=2)
    print(result)


if __name__ == "__main__":
    main()
