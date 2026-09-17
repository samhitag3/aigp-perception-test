#!/usr/bin/env python3
"""Preview training-only union-mask corruption without modifying source data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatenet.utils import load_config
from gateposenet.mask_noise import WindowMaskNoise


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda x: int(x.get("frame_index", 0)))
    return rows


def mask_path(seq_dir: Path, frame: dict) -> Path | None:
    rel = (frame.get("files") or {}).get("instance_mask")
    return seq_dir / rel if rel else None


def to_bgr(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(((mask > 0).astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--sequence-id", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--force-noisy", action="store_true",
                    help="set clean_window_prob=0 for the preview only")
    args = ap.parse_args()

    cfg = load_config(args.config)
    noise_cfg = dict((cfg.get("data") or {}).get("mask_noise") or {})
    if args.force_noisy:
        noise_cfg["clean_window_prob"] = 0.0
    noise_cfg["enabled"] = True

    root = Path(args.dataset_root).expanduser().resolve()
    seq_dir = root / "sequences" / args.sequence_id
    frames_path = seq_dir / "frames.jsonl"
    if not frames_path.exists():
        raise FileNotFoundError(frames_path)
    rows = read_jsonl(frames_path)
    rows = rows[max(0, args.start): max(0, args.start) + max(1, args.frames)]
    if not rows:
        raise RuntimeError("no frames selected")

    out = Path(args.out_dir)
    (out / "clean").mkdir(parents=True, exist_ok=True)
    (out / "noisy").mkdir(parents=True, exist_ok=True)
    (out / "compare").mkdir(parents=True, exist_ok=True)

    state = WindowMaskNoise(noise_cfg, seed=args.seed)
    writer = None
    meta = {"seed": args.seed, "severity": state.severity,
            "clean_window": state.clean_window, "frames": []}

    for t, fr in enumerate(rows):
        p = mask_path(seq_dir, fr)
        if p is None:
            continue
        ids = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if ids is None:
            raise FileNotFoundError(p)
        if ids.ndim == 3:
            ids = ids[..., 0]
        clean = (ids > 0).astype(np.uint8) * 255
        noisy = state.apply(ids, frame_pos=t)

        clean_bgr, noisy_bgr = to_bgr(clean), to_bgr(noisy)
        H, W = clean.shape
        panel = np.zeros((H + 40, 2 * W, 3), dtype=np.uint8)
        panel[40:, :W] = clean_bgr
        panel[40:, W:] = noisy_bgr
        cv2.putText(panel, "CLEAN INPUT / GT SOURCE", (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(panel, "TRAINING INPUT AFTER NOISE", (W + 10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)

        fi = int(fr.get("frame_index", t))
        cv2.imwrite(str(out / "clean" / f"frame_{fi:06d}.png"), clean)
        cv2.imwrite(str(out / "noisy" / f"frame_{fi:06d}.png"), noisy)
        cv2.imwrite(str(out / "compare" / f"frame_{fi:06d}.jpg"), panel)
        if writer is None:
            writer = cv2.VideoWriter(str(out / "noise_preview.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     args.fps, (2 * W, H + 40))
        writer.write(panel)
        meta["frames"].append({
            "frame_index": fi,
            "clean_pixels": int((clean > 0).sum()),
            "noisy_pixels": int((noisy > 0).sum()),
        })

    if writer is not None:
        writer.release()
    (out / "preview.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "out_dir": str(out),
        "severity": state.severity,
        "clean_window": state.clean_window,
        "frames_written": len(meta["frames"]),
    }, indent=2))


if __name__ == "__main__":
    main()
