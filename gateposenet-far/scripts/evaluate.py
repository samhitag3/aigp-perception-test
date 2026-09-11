#!/usr/bin/env python3
"""Unified evaluation entrypoint for all backends, on the shared test set.

GateNet:
    python scripts/evaluate.py --model gatenet \
        --checkpoint runs/gatenet_synth/best.pt --config configs/gatenet_synth.yaml

YOLO26 / YOLOE:
    python scripts/evaluate.py --model yolo26 --weights yolo26/.../best.pt
    python scripts/evaluate.py --model yoloe  --weights yolo26e/.../best.pt

All report a common binary-mask IoU/Dice/PR/F1 over data/test; YOLO also reports
box/mask mAP. Use --json to dump machine-readable results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finetune.registry import BACKENDS


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=BACKENDS)
    ap.add_argument("--test-dir", default="data/test")
    ap.add_argument("--device", default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    # gatenet
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--config", default=None)
    # yolo
    ap.add_argument("--weights", default=None)
    ap.add_argument("--data", default="data/yolo/data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--no-native", action="store_true", help="skip YOLO mAP")
    ap.add_argument("--json", default=None, help="write results to this JSON path")
    args = ap.parse_args()

    if args.model == "gatenet":
        if not (args.checkpoint and args.config):
            raise SystemExit("gatenet eval requires --checkpoint and --config")
        from evaluation.gatenet_eval import evaluate
        res = evaluate(args.checkpoint, args.config, args.test_dir, args.device,
                       args.threshold)
    else:
        if not args.weights:
            raise SystemExit(f"{args.model} eval requires --weights")
        from evaluation.yolo_eval import evaluate
        res = evaluate(args.weights, args.data, args.test_dir, family=args.model,
                       imgsz=args.imgsz, device=args.device, native=not args.no_native)

    print("\n=== evaluation ===")
    for k, v in res.items():
        print(f"  {k:14s}: {v:.4f}" if isinstance(v, float) else f"  {k:14s}: {v}")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
