#!/usr/bin/env python3
import argparse
from gatesynth.config import load_config
from gatesynth.generator import generate_dataset

p = argparse.ArgumentParser(description="Generate canonical synthetic UAV gate sequences.")
p.add_argument("--config", required=True)
p.add_argument("--gate-skin", required=True, help="Path to SAMPLE_GATE_aigp.jpg")
p.add_argument("--output", default=None)
p.add_argument("--workers", type=int, default=None, help="Override generation.num_workers")
p.add_argument("--sequences", type=int, default=None, help="Override sequences.count")
p.add_argument("--seed", type=int, default=None, help="Override global random seed")
p.add_argument("--photometric-backend", default=None, choices=["cpu", "auto", "torch", "cuda", "gpu"], help="Override acceleration.photometric_backend")
p.add_argument("--device", default=None, help="Override acceleration.device, e.g. cpu or cuda")
p.add_argument("--frame-batch-size", type=int, default=None, help="Override acceleration.frame_batch_size")
a = p.parse_args()

cfg = load_config(a.config)
if a.workers is not None:
    cfg.setdefault("generation", {})["num_workers"] = int(a.workers)
if a.sequences is not None:
    cfg.setdefault("sequences", {})["count"] = int(a.sequences)
if a.seed is not None:
    cfg["seed"] = int(a.seed)
if a.photometric_backend is not None:
    cfg.setdefault("acceleration", {})["photometric_backend"] = a.photometric_backend
if a.device is not None:
    cfg.setdefault("acceleration", {})["device"] = a.device
if a.frame_batch_size is not None:
    cfg.setdefault("acceleration", {})["frame_batch_size"] = int(a.frame_batch_size)

root = generate_dataset(cfg, a.gate_skin, a.output)
print(root)
