#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatenet.utils import load_checkpoint, load_config, pick_device
from gateposenet.model import build_gateposenet_mg


OUTPUT_NAMES = [
    "presence_logit",
    "target_logit",
    "corners_uv",
    "corner_inside_logit",
    "center_uv",
    "position",
    "log_depth",
    "rot6d",
    "visible_frac",
    "mask_logit",
    "h5_out",
    "h4_out",
]


class MaskPoseNetStep(torch.nn.Module):
    """
    Stateful one-frame MaskPoseNet deployment graph.

    Inputs:
        mask_image : [1, 3, H, W]
        ego        : [1, 7]
        h5_in      : [1, C5, H/16, W/16]
        h4_in      : [1, C4, H/8, W/8]

    Outputs:
        regular MaskPoseNet per-query outputs
        + updated recurrent states h5_out/h4_out
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        mask_image,
        ego,
        h5_in,
        h4_in,
    ):
        out, hidden = self.model.step(
            mask_image,
            ego,
            (h5_in, h4_in),
        )

        h5_out, h4_out = hidden

        return (
            out["presence_logit"],
            out["target_logit"],
            out["corners_uv"],
            out["corner_inside_logit"],
            out["center_uv"],
            out["position"],
            out["log_depth"],
            out["rot6d"],
            out["visible_frac"],
            out["mask_logit"],
            h5_out,
            h4_out,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    cfg = load_config(args.config)
    device = pick_device(args.device)

    H = int(cfg["data"].get("height", 192))
    W = int(cfg["data"].get("width", 320))

    model = build_gateposenet_mg(cfg["model"]).to(device).eval()

    ckpt = load_checkpoint(
        args.checkpoint,
        model,
        map_location=device,
    )

    wrapper = MaskPoseNetStep(model).to(device).eval()

    # ----------------------------------------------------------
    # Dummy deployment inputs
    # ----------------------------------------------------------

    mask_image = torch.zeros(
        1,
        3,
        H,
        W,
        dtype=torch.float32,
        device=device,
    )

    # [vx, vy, vz, wx, wy, wz, dt]
    ego = torch.zeros(
        1,
        7,
        dtype=torch.float32,
        device=device,
    )

    # 60 FPS camera
    ego[:, 6] = 1.0 / 60.0

    hidden = model.init_hidden(
        batch=1,
        hw=(H, W),
        device=device,
    )

    if not isinstance(hidden, tuple):
        raise RuntimeError(
            "This exporter expects temporal_highres=true "
            "and therefore (h5, h4) hidden states."
        )

    h5, h4 = hidden

    print("\nINPUTS")
    print("------")
    print("mask_image:", tuple(mask_image.shape))
    print("ego       :", tuple(ego.shape))
    print("h5_in     :", tuple(h5.shape))
    print("h4_in     :", tuple(h4.shape))

    with torch.inference_mode():
        outputs = wrapper(
            mask_image,
            ego,
            h5,
            h4,
        )

    print("\nOUTPUTS")
    print("-------")

    for name, tensor in zip(OUTPUT_NAMES, outputs):
        print(f"{name:24s}: {tuple(tensor.shape)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nExporting ONNX...")

    torch.onnx.export(
        wrapper,
        (
            mask_image,
            ego,
            h5,
            h4,
        ),
        str(output_path),

        input_names=[
            "mask_image",
            "ego",
            "h5_in",
            "h4_in",
        ],

        output_names=OUTPUT_NAMES,

        opset_version=17,

        # Static shapes are intentional for TensorRT.
        dynamo=False,

        do_constant_folding=True,
    )

    print(f"\nExported -> {output_path}")

    print(
        f"Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB"
    )

    # ----------------------------------------------------------
    # ONNX checker
    # ----------------------------------------------------------

    try:
        import onnx

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

        print("onnx.checker: OK")

    except Exception as exc:
        print(f"ONNX checker skipped/failed: {exc}")


if __name__ == "__main__":
    main()