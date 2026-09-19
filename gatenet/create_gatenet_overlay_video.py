#!/usr/bin/env python3
"""Create an RGB video with translucent GateNet segmentation overlays."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def natural_key(value: str) -> list[object]:
    return [int(piece) if piece.isdigit() else piece.lower()
            for piece in re.split(r"(\d+)", value)]


def index_images(directory: Path) -> dict[str, Path]:
    images = {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    if not images:
        raise RuntimeError(f"No images found in {directory}")
    return images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--mask-threshold", type=float, default=127.0)
    parser.add_argument(
        "--color",
        type=int,
        nargs=3,
        metavar=("B", "G", "R"),
        default=(0, 255, 0),
        help="Overlay color in OpenCV BGR order; default is green.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")

    rgb_images = index_images(args.rgb_dir)
    mask_images = index_images(args.mask_dir)
    stems = sorted(rgb_images.keys() & mask_images.keys(), key=natural_key)
    missing_masks = sorted(rgb_images.keys() - mask_images.keys(), key=natural_key)

    if not stems:
        raise RuntimeError("No RGB and mask filenames share the same stem")
    if missing_masks:
        preview = ", ".join(missing_masks[:10])
        raise RuntimeError(
            f"Missing masks for {len(missing_masks)} RGB frames; first: {preview}"
        )

    first_frame = cv2.imread(str(rgb_images[stems[0]]), cv2.IMREAD_COLOR)
    if first_frame is None:
        raise RuntimeError(f"Could not read {rgb_images[stems[0]]}")
    height, width = first_frame.shape[:2]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {args.output}")

    color = np.asarray(args.color, dtype=np.float32)
    try:
        for index, stem in enumerate(stems, 1):
            frame = cv2.imread(str(rgb_images[stem]), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_images[stem]), cv2.IMREAD_UNCHANGED)
            if frame is None or mask is None:
                raise RuntimeError(f"Could not read RGB frame or mask for {stem}")

            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            if mask.ndim == 3:
                mask = np.max(mask, axis=2)
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

            foreground = mask > args.mask_threshold
            output_frame = frame.copy()
            if np.any(foreground):
                source = frame[foreground].astype(np.float32)
                blended = (1.0 - args.alpha) * source + args.alpha * color
                output_frame[foreground] = np.clip(blended, 0, 255).astype(np.uint8)

                contours, _ = cv2.findContours(
                    foreground.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(output_frame, contours, -1, tuple(args.color), 1)

            cv2.putText(
                output_frame,
                f"frame {index - 1:06d}",
                (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(output_frame)

            if index % 100 == 0 or index == len(stems):
                print(f"[{index}/{len(stems)}] {stem}")
    finally:
        writer.release()

    duration = len(stems) / args.fps
    print(f"Wrote: {args.output}")
    print(f"Frames: {len(stems)}")
    print(f"FPS: {args.fps:g}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Resolution: {width}x{height}")


if __name__ == "__main__":
    main()
