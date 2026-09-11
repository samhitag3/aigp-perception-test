from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--out", default="data/synth")
    parser.add_argument("--prefix", default="isaac")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--min-mask-pixels", type=int, default=64)
    parser.add_argument("--motion-blur-prob", type=float, default=0.25)
    parser.add_argument("--noise-std", type=float, default=4.0)
    parser.add_argument("--brightness-jitter", type=float, default=0.12)
    parser.add_argument("--jpeg-quality", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def read_labels(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def labels_contain_gate(label_obj) -> bool:
    if isinstance(label_obj, dict):
        class_value = str(label_obj.get("class", ""))
        tokens = [part.strip().lower() for part in class_value.split(",")]
        return "gate" in tokens

    text = str(label_obj).lower()
    return "gate" in text


def gate_ids_from_labels(labels: dict) -> list[int]:
    ids = []
    for raw_id, label_obj in labels.items():
        if labels_contain_gate(label_obj):
            ids.append(int(raw_id))
    return ids


def read_rgb(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"failed to read RGB image: {path}")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img


def read_semantic_ids(stem: str, semantic_dir: Path):
    npy_path = semantic_dir / f"semantic_{stem}.npy"
    if npy_path.exists():
        return np.load(npy_path).astype(np.uint32)

    png_path = semantic_dir / f"semantic_{stem}.png"
    sem = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if sem is None:
        raise RuntimeError(f"failed to read semantic IDs: {png_path}")

    if sem.ndim == 3:
        raise RuntimeError(
            f"{png_path} looks colorized. Expected raw semantic IDs. "
            "Make sure the Isaac writer uses colorize=False."
        )

    return sem.astype(np.uint32)


def motion_blur(img, rng):
    k = int(rng.choice([3, 5, 7, 9, 11]))
    angle = float(rng.uniform(0.0, 180.0))

    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0

    center = (k / 2 - 0.5, k / 2 - 0.5)
    rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (k, k))
    kernel /= max(kernel.sum(), 1e-6)

    return cv2.filter2D(img, -1, kernel)


def augment_image(img, rng, args):
    out = img.astype(np.float32)

    if args.brightness_jitter > 0:
        alpha = 1.0 + rng.uniform(-args.brightness_jitter, args.brightness_jitter)
        beta = rng.uniform(-18.0, 18.0)
        out = out * alpha + beta

    if args.noise_std > 0:
        out += rng.normal(0.0, args.noise_std, out.shape)

    out = np.clip(out, 0, 255).astype(np.uint8)

    if rng.random() < args.motion_blur_prob:
        out = motion_blur(out, rng)

    if args.jpeg_quality and args.jpeg_quality < 100:
        quality = int(np.clip(args.jpeg_quality, 1, 100))
        ok, encoded = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            out = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    return out


def next_free_stem(out_dir: Path, prefix: str, start: int, overwrite: bool):
    idx = start
    while True:
        stem = f"{prefix}_{idx:06d}"
        image_path = out_dir / "images" / f"{stem}.png"
        mask_path = out_dir / "masks" / f"{stem}.png"

        if overwrite or (not image_path.exists() and not mask_path.exists()):
            return stem, idx + 1

        idx += 1


def main():
    args = parse_args()
    raw = Path(args.raw)
    out = Path(args.out)

    rgb_dir = raw / "rgb"
    semantic_dir = raw / "semantic"

    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"missing RGB directory: {rgb_dir}")
    if not semantic_dir.is_dir():
        raise FileNotFoundError(f"missing semantic directory: {semantic_dir}")

    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    rgb_paths = sorted(rgb_dir.glob("rgb_*.png"))
    if not rgb_paths:
        raise RuntimeError(f"no rgb_*.png files found in {rgb_dir}")

    rng = np.random.default_rng(args.seed)

    written = 0
    skipped = 0
    next_idx = args.start_index

    for rgb_path in rgb_paths:
        raw_stem = rgb_path.stem.replace("rgb_", "")
        labels_path = semantic_dir / f"semantic_labels_{raw_stem}.json"

        if not labels_path.exists():
            print(f"skip {rgb_path.name}: missing {labels_path.name}")
            skipped += 1
            continue

        labels = read_labels(labels_path)
        gate_ids = gate_ids_from_labels(labels)

        if not gate_ids:
            print(f"skip {rgb_path.name}: no semantic id labeled gate")
            skipped += 1
            continue

        img = read_rgb(rgb_path)
        sem = read_semantic_ids(raw_stem, semantic_dir)

        mask = np.isin(sem, gate_ids).astype(np.uint8) * 255

        if int((mask > 0).sum()) < args.min_mask_pixels:
            skipped += 1
            continue

        img = augment_image(img, rng, args)

        out_stem, next_idx = next_free_stem(out, args.prefix, next_idx, args.overwrite)
        image_out = out / "images" / f"{out_stem}.png"
        mask_out = out / "masks" / f"{out_stem}.png"

        cv2.imwrite(str(image_out), img)
        cv2.imwrite(str(mask_out), mask)

        written += 1

    print(f"wrote {written} samples to {out}")
    print(f"skipped {skipped} raw frames")


if __name__ == "__main__":
    main()
    