#!/usr/bin/env python3
"""Export GateNet to ONNX for onboard deployment (Jetson Orin / TensorRT).

Only the highest-resolution output (y0) is exported, wrapped in
GateNetInference so the graph outputs a sigmoid probability map (optionally
resized / thresholded).

Usage:
    python scripts/export_onnx.py --config configs/gatenet_a2rl.yaml \
        --checkpoint runs/gatenet_a2rl/best.pt --output gatenet.onnx

TensorRT (on the Jetson):
    trtexec --onnx=gatenet.onnx --saveEngine=gatenet.plan --fp16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatenet.model import GateNetInference, build_gatenet
from gatenet.utils import load_checkpoint, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", default="gatenet.onnx")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--binary", action="store_true",
                    help="threshold the output to a binary mask in-graph")
    args = ap.parse_args()

    cfg = load_config(args.config)
    size = (cfg["data"]["height"], cfg["data"]["width"])
    infer_cfg = cfg.get("infer", {})

    net = build_gatenet(cfg["model"])
    load_checkpoint(args.checkpoint, net, map_location="cpu")
    model = GateNetInference(
        net,
        out_size=infer_cfg.get("out_size"),
        threshold=infer_cfg.get("threshold", 0.5) if args.binary else None,
    ).eval()

    dummy = torch.randn(1, cfg["model"].get("in_channels", 3), *size)
    torch.onnx.export(
        model, dummy, args.output,
        input_names=["image"], output_names=["mask"],
        opset_version=args.opset,
        dynamic_axes={"image": {0: "batch"}, "mask": {0: "batch"}},
    )
    print(f"exported -> {args.output}  (input 1x{cfg['model'].get('in_channels',3)}x{size[0]}x{size[1]})")

    try:
        import onnx
        onnx.checker.check_model(onnx.load(args.output))
        print("onnx.checker: OK")
    except Exception as e:  # pragma: no cover
        print(f"onnx check skipped/failed: {e}")


if __name__ == "__main__":
    main()
