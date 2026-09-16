#!/usr/bin/env python3
"""Unified fine-tuning entrypoint for all backends.

GateNet (lightweight U-Net, in-repo):
    python scripts/finetune.py --model gatenet \
        --config configs/gatenet_finetune.yaml --pretrained runs/gatenet_a2rl/best.pt
    # optional: --freeze-encoder  --reset-heads  --freeze-bn-stats

YOLO26 instance segmentation (Ultralytics):
    python scripts/finetune.py --model yolo26 --variant s \
        --data data/yolo/data.yaml --epochs 100 --imgsz 640

YOLOE open-vocabulary segmentation (Ultralytics):
    python scripts/finetune.py --model yoloe --variant 26s \
        --data data/yolo/data.yaml --epochs 80 --imgsz 640

Datasets come from scripts/import_data.py (sam3-autolabeler -> data/).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finetune.backends import get_backend
from finetune.registry import BACKENDS


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=BACKENDS)
    ap.add_argument("--device", default=None, help="cuda/cpu/0; default auto (GPU 0 for YOLO)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--output", default=None, help="output/project dir override")
    ap.add_argument("--name", default=None, help="run name (YOLO)")
    ap.add_argument("--seed", type=int, default=0)

    g = ap.add_argument_group("gatenet")
    g.add_argument("--config", default=None)
    g.add_argument("--pretrained", default=None)
    g.add_argument("--freeze-encoder", action="store_true")
    g.add_argument("--reset-heads", action="store_true")
    g.add_argument("--freeze-bn-stats", action="store_true")

    y = ap.add_argument_group("yolo26 / yoloe")
    y.add_argument("--data", default="data/yolo/data.yaml", help="YOLO data.yaml")
    y.add_argument("--variant", default=None, help="size variant (e.g. s, 26s)")
    y.add_argument("--weights", default=None, help="override base checkpoint")
    y.add_argument("--imgsz", type=int, default=None)
    y.add_argument("--batch", type=int, default=None)
    y.add_argument("--patience", type=int, default=20)
    y.add_argument("--classes", nargs="*", default=["gate"], help="YOLOE class names")
    return ap


def main():
    args = build_parser().parse_args()
    backend = get_backend(args.model)
    result = backend.finetune(args)
    print("\n=== fine-tune result ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
