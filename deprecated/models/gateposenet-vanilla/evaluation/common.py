"""Shared helpers for evaluation: load a GateNet-format test set."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def load_test_pairs(test_dir):
    """Return [(stem, image_path, mask_path)] from <test_dir>/images + /masks."""
    test_dir = Path(test_dir)
    img_dir, msk_dir = test_dir / "images", test_dir / "masks"
    if not img_dir.is_dir() or not msk_dir.is_dir():
        raise FileNotFoundError(f"{test_dir} must contain images/ and masks/")
    pairs = []
    for ip in sorted(img_dir.iterdir()):
        if ip.suffix.lower() not in IMG_EXTS:
            continue
        mp = msk_dir / f"{ip.stem}.png"
        if mp.exists():
            pairs.append((ip.stem, ip, mp))
    if not pairs:
        raise RuntimeError(f"no image/mask pairs in {test_dir}")
    return pairs


def read_gt_mask(path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return (m > 127).astype(np.uint8)
