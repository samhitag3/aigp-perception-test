#!/usr/bin/env python3
"""Benchmark the GatePoseNet deployment step on whatever runtime is available.

Measures the FULL per-frame cost — preprocess (resize/normalize) + inference
+ hidden-state feedback + postprocess — which is what the 90-120 Hz budget is
spent on. Feeds a realistic 640x360 BGR frame (the AIGP camera resolution).

    uv run python -m deploy.benchmark --model runs/gateposenet_traj/gateposenet_step.onnx
    python3 -m deploy.benchmark --model gatepose.plan          # on the Jetson
    uv run python -m deploy.benchmark --model x.onnx --runtime both   # TRT vs ORT

Reports mean / p50 / p95 / max latency and the implied rate over --iters
steps (default 300) after --warmup steps (default 30).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from deploy.trt_runtime import (GatePoseORT, GatePoseTRT, _tensorrt_available,
                                build_engine, make_runtime)


def bench(rt, frame: np.ndarray, ego: np.ndarray, iters: int,
          warmup: int) -> dict:
    rt.reset()
    for _ in range(warmup):
        rt.step(frame, ego)
    lat = np.empty(iters)
    for i in range(iters):
        t0 = time.perf_counter()
        rt.step(frame, ego)
        lat[i] = time.perf_counter() - t0
    lat *= 1e3
    return {
        "mean_ms": float(lat.mean()), "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)), "max_ms": float(lat.max()),
        "hz": 1000.0 / float(lat.mean()),
    }


def _label(rt) -> str:
    if isinstance(rt, GatePoseTRT):
        return f"TensorRT ({Path(rt.model_path).suffix}, {rt.cuda.kind})"
    return f"onnxruntime ({rt.provider})"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True,
                    help=".onnx or TensorRT engine (.plan/.engine/.trt)")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--runtime", choices=["auto", "trt", "ort", "both"],
                    default="auto")
    ap.add_argument("--frame-size", default="640x360",
                    help="input BGR frame WxH fed to step() (default 640x360)")
    args = ap.parse_args()

    w, h = (int(v) for v in args.frame_size.split("x"))
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    ego = np.zeros(7, np.float32)
    ego[6] = 1.0 / 120.0

    path = Path(args.model)
    runtimes = []
    if args.runtime in ("auto",):
        runtimes.append(make_runtime(path, args.manifest))
    if args.runtime in ("trt", "both"):
        if path.suffix == ".onnx":
            if not _tensorrt_available():
                raise SystemExit("tensorrt not available for --runtime trt")
            plan = path.with_suffix(".plan")
            if not plan.exists() or \
                    plan.stat().st_mtime < path.stat().st_mtime:
                build_engine(path, plan, fp16=True)
            runtimes.append(GatePoseTRT(plan, args.manifest))
        else:
            runtimes.append(GatePoseTRT(path, args.manifest))
    if args.runtime in ("ort", "both"):
        onnx = path if path.suffix == ".onnx" else path.with_suffix(".onnx")
        runtimes.append(GatePoseORT(onnx, args.manifest))

    rows = []
    for rt in runtimes:
        label = _label(rt)
        print(f"benchmarking {label}: {args.iters} iters "
              f"(+{args.warmup} warmup), frame {w}x{h} -> "
              f"{rt.input_wh[0]}x{rt.input_wh[1]}, model={rt.model_type}")
        rows.append((label, bench(rt, frame, ego, args.iters, args.warmup)))

    hdr = f"{'runtime':<38} {'mean ms':>8} {'p50 ms':>8} {'p95 ms':>8} " \
          f"{'max ms':>8} {'Hz':>7}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for label, r in rows:
        print(f"{label:<38} {r['mean_ms']:>8.2f} {r['p50_ms']:>8.2f} "
              f"{r['p95_ms']:>8.2f} {r['max_ms']:>8.2f} {r['hz']:>7.0f}")
    print("\nbudget: 90-120 Hz -> 8.3-11.1 ms per step (leave margin for "
          "capture/decode + downstream)")


if __name__ == "__main__":
    main()
