#!/usr/bin/env python3
"""Import datasets from the sam3-autolabeler into gateNET-1/data.

Discovers every autolabeler dataset (images/ + labels/ + masks/) under the
source root and writes a single split dataset in BOTH formats:

    data/yolo/{train,val,test}/{images,labels} + data/yolo/data.yaml  (YOLO26/YOLOE)
    data/synth|real/{images,masks}                                    (GateNet pool)
    data/test/{images,masks}                                          (GateNet test)
    data/meta/{intrinsics.json, import_manifest.json}

Frames are symlinked by default (no duplication). Use --copy to copy.

Examples:
    python scripts/import_data.py                       # src=/Users/c.k./sam3-autolabeler
    python scripts/import_data.py --limit-per-ds 200 --copy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finetune.data import DEFAULT_OUT, DEFAULT_SRC, import_all


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", nargs="+", default=[DEFAULT_SRC],
                    help="source root(s) to scan for autolabeler datasets")
    ap.add_argument("--out", default=DEFAULT_OUT, help="destination data/ dir")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--link", action="store_true",
                    help="symlink instead of copying (saves space, but breaks if "
                         "the source is later moved/cleaned — copy is the default)")
    ap.add_argument("--limit-per-ds", type=int, default=0, help="cap frames per dataset")
    args = ap.parse_args()

    manifest = import_all(src_roots=args.src, out=args.out, val_frac=args.val_frac,
                          test_frac=args.test_frac, seed=args.seed, copy=not args.link,
                          limit_per_ds=args.limit_per_ds)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
