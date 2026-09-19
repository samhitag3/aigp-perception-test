#!/usr/bin/env python3
"""Prepare deterministic sequence manifests for vanilla GatePoseNet experiments.

Rules:
- Existing official train/validation/test manifests are preserved if all exist.
- If none exist, --create-base-if-missing creates a deterministic 80/10/10
  sequence split using a stable SHA-256 ordering seeded by --seed.
- Cheap and baseline/tuning subsets are deterministic nested subsets of the
  official TRAIN split only. Validation/test are never subsampled here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def read_manifest(path: Path) -> list[str]:
    return [x.strip() for x in path.read_text().splitlines()
            if x.strip() and not x.lstrip().startswith("#")]


def stable_order(ids: list[str], seed: int) -> list[str]:
    def key(sid: str):
        return hashlib.sha256(f"{seed}\0{sid}".encode("utf-8")).digest()
    return sorted(ids, key=key)


def write_manifest(path: Path, ids: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{x}\n" for x in ids), encoding="utf-8")


def ensure_disjoint(train: list[str], val: list[str], test: list[str]):
    a, b, c = set(train), set(val), set(test)
    if a & b or a & c or b & c:
        raise RuntimeError("train/validation/test manifests overlap by sequence")


def subset_count(n: int, frac: float) -> int:
    if n <= 0:
        return 0
    return min(n, max(1, int(math.floor(n * frac))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cheap-fraction", type=float, default=0.15)
    ap.add_argument("--baseline-fraction", type=float, default=0.50)
    ap.add_argument("--create-base-if-missing", action="store_true")
    ap.add_argument("--force-rebuild-derived", action="store_true")
    args = ap.parse_args()

    if not (0 < args.cheap_fraction <= args.baseline_fraction <= 1.0):
        raise ValueError("require 0 < cheap_fraction <= baseline_fraction <= 1")

    root = Path(args.dataset_root).expanduser().resolve()
    seq_dir = root / "sequences"
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"missing sequences directory: {seq_dir}")
    all_ids = sorted(p.name for p in seq_dir.iterdir() if p.is_dir())
    if not all_ids:
        raise RuntimeError(f"no sequence directories found under {seq_dir}")

    splits = root / "splits"
    p_train = splits / "train_sequences.txt"
    p_val = splits / "validation_sequences.txt"
    p_val_alias = splits / "val_sequences.txt"
    p_test = splits / "test_sequences.txt"
    # Older generated datasets sometimes used val_sequences.txt. Normalize that
    # alias once so every model can use the canonical validation_sequences name.
    if p_val_alias.exists():
        if p_val.exists():
            if read_manifest(p_val) != read_manifest(p_val_alias):
                raise RuntimeError("validation_sequences.txt and val_sequences.txt disagree")
        else:
            p_val.write_text(p_val_alias.read_text(), encoding="utf-8")
            print(f"normalized legacy val manifest -> {p_val}")
    exists = [p.exists() for p in (p_train, p_val, p_test)]

    if all(exists):
        train, val, test = map(read_manifest, (p_train, p_val, p_test))
        print("preserving existing official train/validation/test manifests")
    elif any(exists):
        raise RuntimeError(
            "only some official split manifests exist; refusing to guess. "
            "Create/fix all three before continuing.")
    else:
        if not args.create_base_if_missing:
            raise FileNotFoundError(
                "official split manifests are missing. Re-run with "
                "--create-base-if-missing to create deterministic 80/10/10 sequence splits.")
        ordered = stable_order(all_ids, args.seed)
        n = len(ordered)
        n_train = int(math.floor(0.80 * n))
        n_val = int(math.floor(0.10 * n))
        # Ensure nonempty splits whenever there are enough sequences.
        if n >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            if n_train + n_val >= n:
                n_train = max(1, n - 2)
                n_val = 1
        train = ordered[:n_train]
        val = ordered[n_train:n_train + n_val]
        test = ordered[n_train + n_val:]
        write_manifest(p_train, train)
        write_manifest(p_val, val)
        write_manifest(p_test, test)
        print(f"created official sequence split: train={len(train)} val={len(val)} test={len(test)}")

    ensure_disjoint(train, val, test)
    all_set = set(all_ids)
    unknown = (set(train) | set(val) | set(test)) - all_set
    if unknown:
        raise RuntimeError(f"split manifests reference missing sequences: {sorted(unknown)[:10]}")

    ranked_train = stable_order(train, args.seed)
    n_cheap = subset_count(len(ranked_train), args.cheap_fraction)
    n_baseline = subset_count(len(ranked_train), args.baseline_fraction)

    # Reuse a legacy seed-42 cheap manifest when present. Earlier GatePoseNet
    # experiments used splits/train_cheap_sequences.txt; preserving those exact
    # sequence IDs makes new vanilla runs directly comparable to those runs.
    legacy_cheap = splits / "train_cheap_sequences.txt"
    cheap_origin = "stable_sha256_seed42"
    if legacy_cheap.exists() and not args.force_rebuild_derived:
        candidate = read_manifest(legacy_cheap)
        if not set(candidate).issubset(set(train)):
            raise RuntimeError(f"legacy cheap manifest is not a subset of official train: {legacy_cheap}")
        if len(candidate) != n_cheap:
            raise RuntimeError(
                f"legacy cheap manifest has {len(candidate)} sequences but expected {n_cheap} "
                f"for cheap_fraction={args.cheap_fraction}; use --force-rebuild-derived "
                "to intentionally replace the legacy selection")
        cheap = candidate
        cheap_origin = str(legacy_cheap.relative_to(root))
        print(f"reusing existing cheap manifest for cross-run comparability: {legacy_cheap}")
    else:
        cheap = ranked_train[:n_cheap]

    # Make baseline deterministic AND guarantee cheap is nested inside it even
    # when cheap came from a legacy manifest generated by another seed-42 tool.
    remainder = [sid for sid in ranked_train if sid not in set(cheap)]
    baseline = list(cheap) + remainder[:max(0, n_baseline - len(cheap))]
    tune = list(baseline)

    out_dir = splits / f"vanilla_seed{args.seed}"
    targets = {
        "train_cheap_sequences.txt": cheap,
        "train_baseline_sequences.txt": baseline,
        "train_tune_sequences.txt": tune,
    }
    for name, ids in targets.items():
        path = out_dir / name
        if path.exists() and not args.force_rebuild_derived:
            old = read_manifest(path)
            if old != ids:
                raise RuntimeError(
                    f"existing derived manifest differs from deterministic result: {path}. "
                    "Use --force-rebuild-derived only if you intentionally want to replace it.")
        else:
            write_manifest(path, ids)

    info = {
        "schema_version": "1.0.0",
        "seed": args.seed,
        "ordering_algorithm": "sha256(seed + NUL + sequence_id)",
        "official_split_policy": "preserve existing; otherwise deterministic 80/10/10 if explicitly requested",
        "train_sequences": len(train),
        "validation_sequences": len(val),
        "test_sequences": len(test),
        "cheap_fraction": args.cheap_fraction,
        "cheap_sequences": len(cheap),
        "cheap_origin": cheap_origin,
        "baseline_fraction": args.baseline_fraction,
        "baseline_sequences": len(baseline),
        "tuning_manifest": "identical to baseline manifest",
        "nested": set(cheap).issubset(set(baseline)),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    print(f"dataset: {root}")
    print(f"official: train={len(train)} validation={len(val)} test={len(test)}")
    print(f"cheap:    {len(cheap)}/{len(train)} train sequences ({args.cheap_fraction:.0%})")
    print(f"baseline: {len(baseline)}/{len(train)} train sequences ({args.baseline_fraction:.0%})")
    print(f"tune:     {len(tune)}/{len(train)} train sequences (same exact IDs as baseline)")
    print(f"derived manifests -> {out_dir}")


if __name__ == "__main__":
    main()
