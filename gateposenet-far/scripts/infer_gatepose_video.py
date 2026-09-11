#!/usr/bin/env python3
"""Backward-compatible video wrapper for scripts/infer_gatepose.py."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--presence-th", type=float, default=0.5)
    ap.add_argument("--mask-th", type=float, default=0.5)
    ap.add_argument("--overlay-alpha", type=float, default=0.38)
    ap.add_argument("--ego-jsonl", default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--save-overlays", action="store_true",
                    help="retained for compatibility; overlays are now always saved")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    cmd = [
        sys.executable, str(repo_root / "scripts" / "infer_gatepose.py"),
        "--input", args.video,
        "--checkpoint", args.checkpoint,
        "--config", args.config,
        "--out-dir", args.out_dir,
        "--presence-th", str(args.presence_th),
        "--mask-th", str(args.mask_th),
        "--overlay-alpha", str(args.overlay_alpha),
    ]
    if args.device:
        cmd += ["--device", args.device]
    if args.ego_jsonl:
        cmd += ["--ego-jsonl", args.ego_jsonl]
    if args.max_frames is not None:
        cmd += ["--max-frames", str(args.max_frames)]
    if args.show:
        cmd += ["--show"]
    if args.label:
        cmd += ["--label", args.label]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
