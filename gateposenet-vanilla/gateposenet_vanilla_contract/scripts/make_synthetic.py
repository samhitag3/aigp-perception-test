#!/usr/bin/env python3
"""OPTIONAL: dump composited synthetic gate images + masks to disk.

NOTE: Production data should come from the upstream sam3-autolabeler pipeline
(real autolabeled footage + synthetic composites), which emits the standard
images/ + masks/ contract. This script is a lightweight fallback for smoke
tests, CI, or bootstrapping when no labeled footage is available yet.

Usage:
    python scripts/make_synthetic.py --gates data/gate_templates \
        --backgrounds data/backgrounds --out data/synth \
        --n 4000 --height 384 --width 384
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatenet.synth import SyntheticGateDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", required=True)
    ap.add_argument("--backgrounds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--alpha-from", default="alpha", choices=["alpha", "black", "white"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    ds = SyntheticGateDataset(args.gates, args.backgrounds,
                              size=(args.height, args.width), length=args.n,
                              alpha_from=args.alpha_from, seed=args.seed)
    for i in range(args.n):
        img_t, mask_t = ds[i]
        img = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask = (mask_t[0].numpy() * 255).astype(np.uint8)
        stem = f"synth_{i:06d}"
        cv2.imwrite(str(out / "images" / f"{stem}.jpg"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / "masks" / f"{stem}.png"), mask)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{args.n}")
    print(f"wrote {args.n} synthetic samples -> {out}")


if __name__ == "__main__":
    main()
