#!/usr/bin/env python3
"""Export GatePoseNet's single-step deployment graph to ONNX + benchmark it.

The 90-120 Hz perception loop runs `step(image, ego, h) -> (outputs, h)`; this
script exports exactly that stateful step (the recurrent state is an explicit
input/output tensor, so the runtime just feeds it back each frame):

    uv run python scripts/export_gatepose.py \
        --checkpoint runs/gateposenet_traj/best.pt --out gateposenet_step.onnx

Deploy with TensorRT (fp16) on the flight computer:
    trtexec --onnx=gateposenet_step.onnx --fp16 --saveEngine=gatepose.plan

Latency reference (batch 1, 320x192): ~2.4 ms eager fp32 on an RTX A6000
(~425 Hz); TensorRT fp16 on Jetson-class hardware typically lands in the
4-8 ms range — inside the 90-120 Hz budget with margin for capture/decode.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatenet.utils import load_checkpoint, load_config, pick_device
from gateposenet.model_single import \
    build_gateposenet_single as build_gateposenet


class _StepGraph(torch.nn.Module):
    """ONNX-friendly wrapper: fixed output ORDER, explicit hidden state."""

    OUTPUTS = ("corners_uv", "corner_inside_logit", "center_uv", "position",
               "log_depth", "rot6d", "visible_logit", "visible_frac",
               "k_scale", "seg_logit")

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image, ego, h):
        out, h2 = self.net.step(image, ego, h)
        return tuple(out[k] for k in self.OUTPUTS) + (h2,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gateposenet_traj.yaml")
    ap.add_argument("--checkpoint", default="runs/gateposenet_traj/best.pt")
    ap.add_argument("--out", default="runs/gateposenet_traj/gateposenet_step.onnx")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = pick_device(args.device)
    d = cfg["data"]
    h_img, w_img = int(d.get("height", 192)), int(d.get("width", 320))

    net = build_gateposenet(cfg["model"]).to(device).eval()
    load_checkpoint(args.checkpoint, net, map_location=device)
    wrapper = _StepGraph(net).eval()

    image = torch.rand(1, 3, h_img, w_img, device=device)
    ego = torch.zeros(1, 7, device=device)
    ego[0, 6] = 1.0 / 120.0
    hidden = net.init_hidden(1, (h_img, w_img), device=device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, (image, ego, hidden), str(out_path),
        input_names=["image", "ego", "h_in"],
        output_names=list(_StepGraph.OUTPUTS) + ["h_out"],
        opset_version=17, do_constant_folding=True,
    )
    print(f"exported: {out_path} "
          f"({out_path.stat().st_size / 1e6:.1f} MB, opset 17)")

    # eager-latency reference on this machine
    with torch.no_grad():
        h = hidden
        for _ in range(20):
            *_, h = wrapper(image, ego, h)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        n = 300
        for _ in range(n):
            *_, h = wrapper(image, ego, h)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / n * 1000
    print(f"eager step latency here: {ms:.2f} ms ({1000 / ms:.0f} Hz) on "
          f"{device}. Deploy with TensorRT fp16 for the flight computer.")

    # optional onnxruntime parity check
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out_path),
                                    providers=["CPUExecutionProvider"])
        outs = sess.run(None, {
            "image": image.cpu().numpy(), "ego": ego.cpu().numpy(),
            "h_in": hidden.cpu().numpy()})
        ref = wrapper(image, ego, hidden)
        err = max(float(abs(o - r.detach().cpu().numpy()).max())
                  for o, r in zip(outs, ref))
        print(f"onnxruntime parity: max abs err {err:.2e}")
    except ImportError:
        print("onnxruntime not installed — skipped parity check "
              "(uv sync --extra export)")


if __name__ == "__main__":
    main()
