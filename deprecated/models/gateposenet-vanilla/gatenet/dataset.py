"""GateNet datasets and dataloaders.

Data layout (see data/README.md):

    <root>/
      images/   frame_0001.jpg ...        # RGB input images
      masks/    frame_0001.png ...         # binary gate masks (0/255 or 0/1)

Masks must share the image stem (an optional suffix is configurable). The mask
is binarised at load time, so 0/255 PNGs (e.g. from the SAM3 autolabeler) work
directly.

``MergedGateDataset`` mixes a synthetic and a real dataset at a fixed sampling
ratio (MonoRace uses 3500 synthetic : 500 real per epoch).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _list_images(d: Path):
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)


class GateSegDataset(Dataset):
    """A single image/mask folder pair."""

    def __init__(self, root, size, augment=None, mask_suffix: str = ""):
        self.root = Path(root)
        self.img_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        if not self.img_dir.is_dir():
            raise FileNotFoundError(f"{self.img_dir} does not exist")
        self.size = tuple(size)  # (H, W)
        self.augment = augment
        self.mask_suffix = mask_suffix
        self.items = _list_images(self.img_dir)
        if not self.items:
            raise RuntimeError(f"No images found in {self.img_dir}")

    def __len__(self):
        return len(self.items)

    def _find_mask(self, stem: str) -> Path:
        for ext in IMG_EXTS:
            p = self.mask_dir / f"{stem}{self.mask_suffix}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(
            f"No mask for '{stem}' in {self.mask_dir} (suffix='{self.mask_suffix}')")

    def __getitem__(self, idx):
        path = self.items[idx % len(self.items)]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self._find_mask(path.stem)), cv2.IMREAD_GRAYSCALE)

        H, W = self.size
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)

        if self.augment is not None:
            img, mask = self.augment(img, mask)

        img_t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(np.ascontiguousarray(mask))[None]  # (1,H,W)
        return img_t, mask_t


class MergedGateDataset(Dataset):
    """Mix two datasets at a fixed sampling ratio (e.g. synthetic:real).

    Each sample is drawn from the synthetic set with probability
    ratio[0]/sum(ratio), else from the real set. ``length`` controls the nominal
    epoch size (defaults to len(synth) + len(real)).
    """

    def __init__(self, synth: Dataset, real: Dataset, ratio=(3500, 500),
                 length: int | None = None, seed: int = 0):
        self.synth = synth
        self.real = real
        self.p_synth = ratio[0] / float(sum(ratio))
        self.length = length or (len(synth) + len(real))
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        use_synth = self._rng.random() < self.p_synth
        if use_synth and len(self.synth) > 0:
            j = int(self._rng.integers(0, len(self.synth)))
            return self.synth[j]
        if len(self.real) > 0:
            j = int(self._rng.integers(0, len(self.real)))
            return self.real[j]
        # fallback if one side empty
        j = int(self._rng.integers(0, len(self.synth)))
        return self.synth[j]


def split_dataset(ds: Dataset, val_frac: float, seed: int = 42):
    """Random train/val split returning two Subsets."""
    from torch.utils.data import random_split
    n_val = max(1, int(round(len(ds) * val_frac)))
    n_train = len(ds) - n_val
    gen = torch.Generator().manual_seed(seed)
    return random_split(ds, [n_train, n_val], generator=gen)
