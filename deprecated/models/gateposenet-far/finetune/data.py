"""Import datasets from the sam3-autolabeler into gateNET-1/data.

The autolabeler emits, per dataset directory:
    images/   frame_XXXXX.png
    labels/   frame_XXXXX.txt        (YOLO segmentation polygons, class 'gate')
    masks/    frame_XXXXX.png         (union gate-frame mask, 0/255)
              frame_XXXXX_obj_N.png   (per-object masks — skipped)
    data.yaml, ground_truth.json, keypoints.json, ...

This importer discovers every such dataset under one or more source roots and
materialises a single, split dataset under ``data/`` in BOTH formats so any
backend can train on the same frames:

    data/yolo/{train,val,test}/{images,labels}  + data/yolo/data.yaml   (YOLO26/YOLOE)
    data/synth/{images,masks}   data/real/{images,masks}                (GateNet train pool)
    data/test/{images,masks}                                            (GateNet held-out)
    data/meta/{intrinsics.json, import_manifest.json}

The SAME frames land in the test split across formats, so GateNet and YOLO are
evaluated on identical data. Frames are symlinked by default (no duplication;
``data/`` is gitignored); pass copy=True to copy instead.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
OBJ_RE = re.compile(r"_obj_\d+$")

DEFAULT_SRC = "/Users/c.k./sam3-autolabeler"
DEFAULT_OUT = str(Path(__file__).resolve().parent.parent / "data")


@dataclass
class DatasetDir:
    path: Path
    name: str
    is_real: bool


@dataclass
class Frame:
    image: Path
    label: Path | None
    mask: Path | None
    dataset: str
    is_real: bool
    stem: str  # unique stem: "<dataset>__<frame>"


def _is_dataset(d: Path) -> bool:
    return (d / "images").is_dir() and ((d / "labels").is_dir() or (d / "masks").is_dir())


def discover_datasets(roots) -> list[DatasetDir]:
    """Find all autolabeler dataset dirs under the given roots."""
    found, seen = [], set()
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for dirpath, dirnames, _ in os.walk(root):
            d = Path(dirpath)
            # don't descend into the image/label/mask leaves
            dirnames[:] = [x for x in dirnames
                           if x not in ("images", "labels", "masks", "overlays", ".git")]
            if _is_dataset(d) and d not in seen:
                seen.add(d)
                low = str(d).lower()
                # "synthetic_output" contains "output" — exclude synthetic first
                is_real = ("synthetic" not in low) and (
                    "real" in low or "/output/" in low or low.endswith("/output"))
                found.append(DatasetDir(d, d.name, is_real))
    return found


def collect_frames(datasets: list[DatasetDir], limit_per_ds: int = 0) -> list[Frame]:
    frames = []
    for ds in datasets:
        imgs = sorted(p for p in (ds.path / "images").iterdir()
                      if p.suffix.lower() in IMG_EXTS and not OBJ_RE.search(p.stem))
        if limit_per_ds:
            imgs = imgs[:limit_per_ds]
        for ip in imgs:
            stem = ip.stem
            label = ds.path / "labels" / f"{stem}.txt"
            mask = ds.path / "masks" / f"{stem}.png"
            frames.append(Frame(
                image=ip,
                label=label if label.exists() else None,
                mask=mask if mask.exists() else None,
                dataset=ds.name,
                is_real=ds.is_real,
                stem=f"{ds.name}__{stem}",
            ))
    return frames


def _place(src: Path, dst: Path, copy: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def _read_intrinsics(datasets: list[DatasetDir]) -> dict:
    out = {}
    for ds in datasets:
        gt = ds.path / "ground_truth.json"
        if gt.exists():
            try:
                data = json.loads(gt.read_text())
                if "intrinsics" in data:
                    out[ds.name] = data["intrinsics"]
            except Exception:
                pass
    return out


def import_all(src_roots=None, out=DEFAULT_OUT, val_frac=0.1, test_frac=0.1,
               seed=42, copy=True, limit_per_ds=0, class_names=("gate",)) -> dict:
    """Import + split datasets into data/ in YOLO and GateNet formats."""
    src_roots = src_roots or [DEFAULT_SRC]
    out = Path(out)
    datasets = discover_datasets(src_roots)
    if not datasets:
        raise RuntimeError(f"no autolabeler datasets found under {src_roots}")
    frames = collect_frames(datasets, limit_per_ds=limit_per_ds)
    if not frames:
        raise RuntimeError("datasets found but no frames collected")

    rng = random.Random(seed)
    rng.shuffle(frames)
    n = len(frames)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test = frames[:n_test]
    val = frames[n_test:n_test + n_val]
    train = frames[n_test + n_val:]

    # ---- YOLO format (train/val/test) ---------------------------------
    yolo = out / "yolo"
    for split, fs in (("train", train), ("val", val), ("test", test)):
        for fr in fs:
            ext = fr.image.suffix.lower()
            _place(fr.image, yolo / split / "images" / f"{fr.stem}{ext}", copy)
            if fr.label is not None:
                _place(fr.label, yolo / split / "labels" / f"{fr.stem}.txt", copy)
    data_yaml = (
        f"path: {yolo.resolve()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"test: test/images\n"
        f"nc: {len(class_names)}\n"
        f"names: {list(class_names)}\n"
    )
    (yolo).mkdir(parents=True, exist_ok=True)
    (yolo / "data.yaml").write_text(data_yaml)

    # ---- GateNet format (synth/real pools + held-out test) ------------
    def place_mask_pair(fr: Frame, split_dir: Path):
        if fr.mask is None:
            return False
        ext = fr.image.suffix.lower()
        _place(fr.image, split_dir / "images" / f"{fr.stem}{ext}", copy)
        _place(fr.mask, split_dir / "masks" / f"{fr.stem}.png", copy)
        return True

    counts = {"synth": 0, "real": 0, "test": 0}
    for fr in train + val:                      # GateNet does its own val split
        pool = "real" if fr.is_real else "synth"
        if place_mask_pair(fr, out / pool):
            counts[pool] += 1
    for fr in test:
        if place_mask_pair(fr, out / "test"):
            counts["test"] += 1

    # ---- meta ----------------------------------------------------------
    meta = out / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    intrinsics = _read_intrinsics(datasets)
    (meta / "intrinsics.json").write_text(json.dumps(intrinsics, indent=2))
    manifest = {
        "src_roots": [str(r) for r in src_roots],
        "datasets": [{"name": d.name, "path": str(d.path), "is_real": d.is_real}
                     for d in datasets],
        "n_frames": n,
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "gatenet_counts": counts,
        "class_names": list(class_names),
        "linked": not copy,
    }
    (meta / "import_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
