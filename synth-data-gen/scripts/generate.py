#!/usr/bin/env python3
from pathlib import Path
import argparse
from gatesynth.config import load_config
from gatesynth.generator import generate_dataset

p=argparse.ArgumentParser()
p.add_argument("--config",required=True)
p.add_argument("--gate-skin",required=True,help="Path to SAMPLE_GATE_aigp.jpg")
p.add_argument("--output",default=None)
a=p.parse_args()
cfg=load_config(a.config)
root=generate_dataset(cfg,a.gate_skin,a.output)
print(root)
