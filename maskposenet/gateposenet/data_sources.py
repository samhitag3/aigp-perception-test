"""Helpers for one- or multi-source canonical GatePoseNet datasets.

The vanilla network is unchanged.  This module only lets the compatibility
adapter concatenate canonical datasets (e.g. synthetic + Isaac) while keeping
each source's own sequence-level split manifests authoritative.
"""
from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_right
from pathlib import Path
from typing import Iterable

from torch.utils.data import ConcatDataset, Dataset

from .canonical_dataset import CanonicalGateSequenceDataset


@dataclass(frozen=True)
class CanonicalSourceSpec:
    name: str
    root: Path
    train_manifest: str
    val_manifest: str
    test_manifest: str


def canonical_source_specs(data_cfg: dict) -> list[CanonicalSourceSpec]:
    """Normalize legacy single-root and new multi-source config shapes."""
    defaults = {
        "train_manifest": data_cfg.get("train_manifest", "splits/train_sequences.txt"),
        "val_manifest": data_cfg.get("val_manifest", "splits/validation_sequences.txt"),
        "test_manifest": data_cfg.get("test_manifest", "splits/test_sequences.txt"),
    }
    raw_sources = data_cfg.get("sources")
    if raw_sources:
        out: list[CanonicalSourceSpec] = []
        for i, raw in enumerate(raw_sources):
            if not isinstance(raw, dict) or not raw.get("root"):
                raise ValueError(f"data.sources[{i}] must be a mapping with root")
            root = Path(raw["root"]).expanduser().resolve()
            out.append(CanonicalSourceSpec(
                name=str(raw.get("name") or root.name or f"source_{i}"),
                root=root,
                train_manifest=str(raw.get("train_manifest", defaults["train_manifest"])),
                val_manifest=str(raw.get("val_manifest", defaults["val_manifest"])),
                test_manifest=str(raw.get("test_manifest", defaults["test_manifest"])),
            ))
        return out

    root = data_cfg.get("dataset_root")
    if root:
        root_path = Path(root).expanduser().resolve()
        return [CanonicalSourceSpec(
            name=str(data_cfg.get("dataset_name") or root_path.name),
            root=root_path,
            train_manifest=str(defaults["train_manifest"]),
            val_manifest=str(defaults["val_manifest"]),
            test_manifest=str(defaults["test_manifest"]),
        )]
    return []


def _manifest_for(spec: CanonicalSourceSpec, split: str) -> str:
    if split == "train":
        return spec.train_manifest
    if split in {"validation", "val"}:
        return spec.val_manifest
    if split == "test":
        return spec.test_manifest
    raise ValueError(f"unknown split {split!r}")


class MultiSourceCanonicalDataset(ConcatDataset):
    """ConcatDataset with source metadata retained for reporting/debugging."""
    def __init__(self, datasets: list[Dataset], source_specs: list[CanonicalSourceSpec]):
        if not datasets:
            raise ValueError("at least one canonical source is required")
        super().__init__(datasets)
        self.source_specs = list(source_specs)
        self.source_names = [s.name for s in source_specs]
        self.dataset_roots = [str(s.root) for s in source_specs]

    def window_meta(self, index: int) -> dict:
        if index < 0:
            if -index > len(self):
                raise IndexError(index)
            index = len(self) + index
        ds_idx = bisect_right(self.cumulative_sizes, index)
        prev = 0 if ds_idx == 0 else self.cumulative_sizes[ds_idx - 1]
        local_idx = index - prev
        ds = self.datasets[ds_idx]
        meta = dict(ds.window_meta(local_idx)) if hasattr(ds, "window_meta") else {}
        meta["source_name"] = self.source_specs[ds_idx].name
        meta["dataset_root"] = str(self.source_specs[ds_idx].root)
        return meta


def build_canonical_dataset(
    data_cfg: dict,
    split: str,
    *,
    stride: int | None = None,
    ego_dropout: float = 0.0,
    load_masks: bool = True,
) -> Dataset:
    """Build one canonical dataset or a deterministic concatenation of sources."""
    specs = canonical_source_specs(data_cfg)
    if not specs:
        raise ValueError("canonical config must define data.dataset_root or data.sources")

    size = (int(data_cfg.get("height", 192)), int(data_cfg.get("width", 320)))
    if stride is None:
        if split == "train":
            stride = int(data_cfg.get("stride", 4))
        elif split in {"validation", "val"}:
            stride = int(data_cfg.get("val_stride", data_cfg.get("window", 8)))
        else:
            stride = int(data_cfg.get("test_stride", data_cfg.get("window", 8)))

    datasets: list[Dataset] = []
    for spec in specs:
        datasets.append(CanonicalGateSequenceDataset(
            spec.root,
            _manifest_for(spec, split),
            window=int(data_cfg.get("window", 8)),
            stride=int(stride),
            size=size,
            pose_sup_max_m=float(data_cfg.get("pose_sup_max_m", 20.0)),
            nominal_focal=float(data_cfg.get("nominal_focal", 320.0)),
            load_masks=load_masks,
            ego_dropout=float(ego_dropout),
        ))

    if len(datasets) == 1:
        # Attach the same metadata names used by MultiSource for reporting.
        ds = datasets[0]
        ds.source_specs = specs  # type: ignore[attr-defined]
        ds.source_names = [specs[0].name]  # type: ignore[attr-defined]
        ds.dataset_roots = [str(specs[0].root)]  # type: ignore[attr-defined]
        return ds
    return MultiSourceCanonicalDataset(datasets, specs)


def describe_sources(data_cfg: dict) -> list[dict[str, str]]:
    return [
        {
            "name": s.name,
            "root": str(s.root),
            "train_manifest": s.train_manifest,
            "validation_manifest": s.val_manifest,
            "test_manifest": s.test_manifest,
        }
        for s in canonical_source_specs(data_cfg)
    ]
