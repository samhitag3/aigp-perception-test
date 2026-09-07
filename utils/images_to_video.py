#!/usr/bin/env python3

from pathlib import Path
import argparse
import re

import cv2


def natural_sort_key(path: Path):
    """
    Sort filenames naturally:
        frame1.png
        frame2.png
        frame10.png

    instead of:
        frame1.png
        frame10.png
        frame2.png
    """
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Convert a folder of images into an MP4 video."
    )

    parser.add_argument(
        "image_folder",
        type=Path,
        help="Path to folder containing images",
    )

    parser.add_argument(
        "--fps",
        type=float,
        required=True,
        help="Output video FPS",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output MP4 path",
    )

    args = parser.parse_args()

    image_folder = args.image_folder

    if not image_folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {image_folder}")

    if not image_folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {image_folder}")

    # Supported image formats
    extensions = {".jpg", ".jpeg", ".png"}

    image_paths = sorted(
        [
            path
            for path in image_folder.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        ],
        key=natural_sort_key,
    )

    if not image_paths:
        raise RuntimeError(f"No images found in {image_folder}")

    print(f"Found {len(image_paths)} images.")

    # Read first image to determine resolution
    first_frame = cv2.imread(str(image_paths[0]))

    if first_frame is None:
        raise RuntimeError(f"Could not read image: {image_paths[0]}")

    height, width = first_frame.shape[:2]

    # Default output:
    # /path/to/sequence -> /path/to/sequence.mp4
    if args.output is None:
        output_path = image_folder.parent / f"{image_folder.name}.mp4"
    else:
        output_path = args.output

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # MP4 codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        args.fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError("Could not initialize video writer.")

    for i, image_path in enumerate(image_paths):
        frame = cv2.imread(str(image_path))

        if frame is None:
            print(f"Warning: skipping unreadable image: {image_path}")
            continue

        # Ensure every frame has identical resolution
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(
                frame,
                (width, height),
                interpolation=cv2.INTER_AREA,
            )

        writer.write(frame)

        if (i + 1) % 100 == 0 or i + 1 == len(image_paths):
            print(f"{i + 1}/{len(image_paths)} frames written")

    writer.release()

    duration = len(image_paths) / args.fps

    print()
    print(f"Video created: {output_path}")
    print(f"Resolution:    {width}x{height}")
    print(f"FPS:           {args.fps}")
    print(f"Frames:        {len(image_paths)}")
    print(f"Duration:      {duration:.2f} seconds")


if __name__ == "__main__":
    main()