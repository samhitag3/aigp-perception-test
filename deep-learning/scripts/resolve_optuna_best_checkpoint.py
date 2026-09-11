#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import yaml


def main():
    ap = argparse.ArgumentParser(description="Print the best Optuna trial checkpoint path")
    ap.add_argument("--run-dir", required=True, help="Optuna run directory containing study_summary.yaml")
    ap.add_argument("--checkpoint-name", default="best.pt")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    summary_path = run_dir / "study_summary.yaml"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = yaml.safe_load(summary_path.read_text())
    trial = int(summary["best_trial"])
    checkpoint = run_dir / f"trial_{trial:04d}" / args.checkpoint_name
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    print(checkpoint)


if __name__ == "__main__":
    main()
