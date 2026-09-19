#!/usr/bin/env python3
"""Resolve vanilla experiment checkpoints and run visual inference by stage.

Examples:
  uv run python scripts/infer_vanilla_stage.py --regime isaac --stage cheap \
    --input ../data/refined_target/sim0721-10/video.mp4

  uv run python scripts/infer_vanilla_stage.py --regime synth_isaac --stage final \
    --input ../data/refined_target/sim0721-10/images --image-fps 30
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def resolve_artifacts(regime: str, stage: str, stamp: str,
                      runs_root: Path, outputs_root: Path, repo_root: Path):
    config_name = "isaac.yaml" if regime == "isaac" else "combined.yaml"
    base_config = repo_root / "configs" / "vanilla" / config_name

    if stage == "cheap":
        checkpoint = runs_root / f"{regime}_cheap_{stamp}" / "best.pt"
        config = base_config
    elif stage == "baseline":
        checkpoint = runs_root / f"{regime}_baseline_{stamp}" / "best.pt"
        config = base_config
    elif stage == "optuna":
        opt_out = outputs_root / f"{regime}_optuna_{stamp}"
        params_path = opt_out / "best_params.json"
        if not params_path.exists():
            raise FileNotFoundError(
                f"missing {params_path}; finish Optuna first so the best trial can be resolved"
            )
        info = json.loads(params_path.read_text(encoding="utf-8"))
        trial = int(info["best_trial"])
        checkpoint = runs_root / f"{regime}_optuna_{stamp}" / f"trial_{trial:04d}" / "best.pt"
        config = opt_out / "best_config.yaml"
    elif stage == "final":
        checkpoint = runs_root / f"{regime}_final_{stamp}" / "best.pt"
        config = outputs_root / f"{regime}_optuna_{stamp}" / "best_config.yaml"
    else:
        raise ValueError(stage)

    return checkpoint, config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regime", required=True, choices=["isaac", "synth_isaac"])
    ap.add_argument("--stage", required=True, choices=["cheap", "baseline", "optuna", "final"])
    ap.add_argument("--input", required=True)
    ap.add_argument("--stamp", default=datetime.now().strftime("%m%d"))
    ap.add_argument("--runs-root", default="../runs_vanilla")
    ap.add_argument("--outputs-root", default="../outputs_vanilla")
    ap.add_argument("--out-dir", default=None,
                    help="default: ../outputs_vanilla/<regime>_<stage>_infer_<MMDD>")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--presence-th", type=float, default=0.5)
    ap.add_argument("--mask-th", type=float, default=0.5)
    ap.add_argument("--overlay-alpha", type=float, default=0.38)
    ap.add_argument("--image-fps", type=float, default=30.0)
    ap.add_argument("--ego-jsonl", default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--no-overlay-video", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    runs_root = Path(args.runs_root)
    outputs_root = Path(args.outputs_root)
    checkpoint, config = resolve_artifacts(
        args.regime, args.stage, args.stamp, runs_root, outputs_root, repo_root
    )
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint}\n"
            f"Train the {args.regime}/{args.stage} stage first."
        )
    if not config.exists():
        raise FileNotFoundError(f"config not found: {config}")

    out_dir = Path(args.out_dir) if args.out_dir else (
        outputs_root / f"{args.regime}_{args.stage}_infer_{args.stamp}"
    )
    label = f"{args.regime} | {args.stage} | {args.stamp}"

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "infer_gatepose.py"),
        "--input", args.input,
        "--checkpoint", str(checkpoint),
        "--config", str(config),
        "--out-dir", str(out_dir),
        "--device", args.device,
        "--label", label,
        "--presence-th", str(args.presence_th),
        "--mask-th", str(args.mask_th),
        "--overlay-alpha", str(args.overlay_alpha),
        "--image-fps", str(args.image_fps),
    ]
    if args.ego_jsonl:
        cmd += ["--ego-jsonl", args.ego_jsonl]
    if args.max_frames is not None:
        cmd += ["--max-frames", str(args.max_frames)]
    if args.show:
        cmd += ["--show"]
    if args.no_overlay_video:
        cmd += ["--no-overlay-video"]

    print(f"checkpoint: {checkpoint}")
    print(f"config:     {config}")
    print(f"output:     {out_dir}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
