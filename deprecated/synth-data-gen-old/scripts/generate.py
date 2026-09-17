#!/usr/bin/env python3
from pathlib import Path
import argparse
from gatesynth.config import load_config
from gatesynth.generator import generate_dataset

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--gate-skin", required=True, help="Path to SAMPLE_GATE_aigp.jpg")
p.add_argument("--output", default=None)
p.add_argument("--workers", type=int, default=None, help="Override generation.num_workers")
p.add_argument("--photometric-backend", default=None, choices=["cpu", "auto", "torch", "cuda", "gpu"], help="Override acceleration.photometric_backend")
p.add_argument("--device", default=None, help="Override acceleration.device, e.g. cpu or cuda")
a = p.parse_args()

cfg = load_config(a.config)
if a.workers is not None:
    cfg.setdefault("generation", {})["num_workers"] = int(a.workers)
if a.photometric_backend is not None:
    cfg.setdefault("acceleration", {})["photometric_backend"] = a.photometric_backend
if a.device is not None:
    cfg.setdefault("acceleration", {})["device"] = a.device

root = generate_dataset(cfg, a.gate_skin, a.output)
print(root)
