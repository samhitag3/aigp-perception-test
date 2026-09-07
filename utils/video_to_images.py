#!/usr/bin/env python3

from pathlib import Path
import argparse

import cv2


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from a video into a folder of images."
    )

    parser.add_argument(
        "video_path",
        type=Path,
        help="Path to input video",
    )

    parser.add_argument(
        "output_folder",
        type=Path,
        help="Folder where extracted images will be saved",
    )

    parser.add_argument(
        "--file-type",
        type=str,
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="Output image format. Default: png",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="frame",
        help="Filename prefix. Default: frame",
    )

    args = parser.parse_args()

    video_path = args.video_path
    output_folder = args.output_folder
    file_type = args.file_type.lower()

    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    if not video_path.is_file():
        raise RuntimeError(f"Input is not a file: {video_path}")

    output_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video:       {video_path}")
    print(f"Resolution:  {width}x{height}")
    print(f"FPS:         {fps:.3f}")
    print(f"Frames:      {total_frames}")
    print(f"Output:      {output_folder}")
    print(f"Format:      {file_type}")

    # Number of digits for zero-padding.
    # Example:
    # frame_000001.png
    digits = max(6, len(str(max(total_frames, 1))))

    frame_index = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        filename = (
            f"{args.prefix}_{frame_index:0{digits}d}.{file_type}"
        )

        output_path = output_folder / filename

        success = cv2.imwrite(str(output_path), frame)

        if not success:
            cap.release()
            raise RuntimeError(
                f"Failed to write frame: {output_path}"
            )

        frame_index += 1

        if frame_index % 100 == 0:
            if total_frames > 0:
                print(f"{frame_index}/{total_frames} frames extracted")
            else:
                print(f"{frame_index} frames extracted")

    cap.release()

    print()
    print(f"Done. Extracted {frame_index} frames.")
    print(f"Saved to: {output_folder}")


if __name__ == "__main__":
    main()