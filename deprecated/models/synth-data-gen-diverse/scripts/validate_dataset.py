#!/usr/bin/env python3
import argparse, json, sys
from gatesynth.validate import validate_dataset
p=argparse.ArgumentParser(); p.add_argument("dataset_root"); a=p.parse_args()
r=validate_dataset(a.dataset_root); print(json.dumps(r,indent=2)); sys.exit(0 if r["valid"] else 2)
