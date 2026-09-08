#!/usr/bin/env python3
"""Run the standardized gate-pose scorecard over dropped-in datasets.

    # drop dataset dirs (ground_truth.json format) into data/test_drops/, then:
    uv run python scripts/run_gate_tests.py
    uv run python scripts/run_gate_tests.py --datasets path/a path/b

Every dataset is streamed sequence-by-sequence through the model's real
deployment loop (model.step, hidden state carried) and scored on the FIXED
slice grid — maneuver kind, blind vs visible, occlusion level, attitude
(upright/banked/inverted), range buckets, single vs multi gate, target-switch
transitions — so results are comparable across models, datasets and time.
Writes runs/gate_tests/<dataset>/scorecard.{json,md} and prints the tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatenet.utils import load_checkpoint, load_config, pick_device
from gateposenet.model_single import \
    build_gateposenet_single as build_gateposenet
from gatetest.scorecard import (build_scorecard, format_scorecard_md,
                                run_model_on_dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gateposenet_traj.yaml")
    ap.add_argument("--checkpoint", default="runs/gateposenet_traj/best.pt")
    ap.add_argument("--drops-dir", default="data/test_drops")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="explicit dataset dirs (default: every dir with a "
                         "ground_truth.json under --drops-dir)")
    ap.add_argument("--out", default="runs/gate_tests")
    ap.add_argument("--device", default=None)
    ap.add_argument("--fps", type=float, default=None,
                    help="override dt input (default: dataset fps or 30)")
    args = ap.parse_args()

    if args.datasets:
        roots = [Path(d) for d in args.datasets]
    else:
        drops = Path(args.drops_dir)
        roots = sorted(p for p in drops.iterdir()
                       if (p / "ground_truth.json").is_file()) \
            if drops.is_dir() else []
    if not roots:
        raise SystemExit(
            f"no datasets found — drop dataset dirs (with ground_truth.json) "
            f"into {args.drops_dir}/ or pass --datasets")

    cfg = load_config(args.config)
    device = pick_device(args.device)
    d = cfg["data"]
    size = (int(d.get("height", 192)), int(d.get("width", 320)))
    model = build_gateposenet(cfg["model"]).to(device).eval()
    load_checkpoint(args.checkpoint, model, map_location=device)

    for root in roots:
        name = root.resolve().name
        print(f"\n=== {name} ===")
        recs = run_model_on_dataset(model, str(root), device, size=size,
                                    fps_override=args.fps)
        if not recs:
            print("  no scoreable frames — skipped")
            continue
        card = build_scorecard(recs)
        out_dir = Path(args.out) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "scorecard.json", "w") as f:
            json.dump({"dataset": str(root.resolve()),
                       "checkpoint": args.checkpoint,
                       "frames": len(recs), "scorecard": card}, f, indent=2)
        md = format_scorecard_md(card, name)
        (out_dir / "scorecard.md").write_text(md)
        print(md)
        print(f"-> {out_dir}/scorecard.{{json,md}}")


if __name__ == "__main__":
    main()
