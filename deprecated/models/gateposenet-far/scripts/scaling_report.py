#!/usr/bin/env python3
"""GateNet scaling report: parameters, theoretical FLOPs, and inference latency
as a function of the width factor f and input resolution.

GateNet's conv layers have parameter count ~proportional to in_ch * out_ch, and
both scale linearly with f, so total parameters scale as ~f^2. FLOPs scale as
~f^2 * (H*W). This script verifies that empirically and benchmarks latency.

    python scripts/scaling_report.py
    python scripts/scaling_report.py --device cuda --bench-iters 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatenet.model import GateNetInference, build_gatenet


def count_params(f, base=(16, 32, 64, 128, 256)):
    m = build_gatenet({"width_factor": f, "base_channels": list(base)})
    return m.num_parameters(), m


@torch.no_grad()
def bench_latency(model, size, device, iters=20, warmup=5):
    model = model.to(device).eval()
    x = torch.randn(1, 3, size, size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bench-iters", type=int, default=20)
    args = ap.parse_args()
    device = torch.device(args.device)

    print("=" * 68)
    print("PARAMETER SCALING  (params should grow ~ f^2)")
    print("=" * 68)
    print(f"{'f':>4} | {'channels (c1..c5)':>26} | {'params (M)':>10} | {'x vs f=1':>9}")
    print("-" * 68)
    base_params = None
    for f in (1, 2, 4, 8):
        n, m = count_params(f)
        ch = tuple(c for c in [m.inc.block[0].out_channels,
                               m.down1.conv.block[0].out_channels,
                               m.down2.conv.block[0].out_channels,
                               m.down3.conv.block[0].out_channels,
                               m.down4.conv.block[0].out_channels])
        if base_params is None:
            base_params = n
        print(f"{f:>4} | {str(ch):>26} | {n/1e6:>10.3f} | {n/base_params:>8.2f}x")
    print("  (ideal f^2 ratios vs f=1:  1, 4, 16, 64)")

    print()
    print("=" * 68)
    print(f"INFERENCE LATENCY  (device={device}, batch=1, {args.bench_iters} iters)")
    print("=" * 68)
    print(f"{'f':>4} | {'res':>9} | {'params (M)':>10} | {'latency (ms)':>12} | {'FPS':>6}")
    print("-" * 68)
    for f in (2, 4):
        for size in (196, 384):
            n, m = count_params(f)
            wrap = GateNetInference(m)
            ms = bench_latency(wrap, size, device, iters=args.bench_iters)
            print(f"{f:>4} | {size}x{size:<4} | {n/1e6:>10.3f} | {ms:>12.2f} | {1000/ms:>6.1f}")

    print()
    print("=" * 68)
    print("DEEP-SUPERVISION OUTPUT RESOLUTIONS  (f=4, 384x384 input)")
    print("=" * 68)
    m = build_gatenet({"width_factor": 4})
    with torch.no_grad():
        outs = m(torch.randn(1, 3, 384, 384))
    for i, o in enumerate(outs):
        print(f"  y{i}: {tuple(o.shape)}  (1/{384 // o.shape[-1]} resolution)")


if __name__ == "__main__":
    main()
