from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path


def read_split(root: Path, split: str) -> list[str]:
    candidates = [root / "splits" / f"{split}_sequences.txt"]
    if split == "validation":
        candidates.append(root / "splits" / "val_sequences.txt")
    for p in candidates:
        if p.exists():
            return [x.strip() for x in p.read_text().splitlines() if x.strip()]
    raise FileNotFoundError(f"No split manifest for {split} under {root / 'splits'}")


def stable_order(ids: list[str], source_name: str, seed: int) -> list[str]:
    def key(seq_id: str) -> str:
        payload = f"{seed}:{source_name}:{seq_id}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
    return sorted(ids, key=key)


def count_for_fraction(n: int, fraction: float) -> int:
    if n == 0:
        return 0
    if fraction >= 1.0:
        return n
    return max(1, min(n, int(math.floor(n * fraction + 0.5))))


def link_sequence(view_root: Path, source_root: Path, source_name: str, seq_id: str) -> str:
    alias = f"{source_name}__{seq_id}"
    dst = view_root / "sequences" / alias
    src = source_root / "sequences" / seq_id
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    # Absolute symlinks keep the view valid no matter where training is launched from.
    os.symlink(src.resolve(), dst, target_is_directory=True)
    return alias


def make_view(output_root: Path, view_name: str, sources: list[tuple[str, Path]], fraction: float, seed: int) -> dict:
    view = output_root / view_name
    if view.exists():
        shutil.rmtree(view)
    (view / "splits").mkdir(parents=True, exist_ok=True)
    (view / "sequences").mkdir(parents=True, exist_ok=True)

    split_aliases: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    meta_sources = []
    for source_name, root in sources:
        source_meta = {"name": source_name, "root": str(root.resolve()), "splits": {}}
        for split in ("train", "validation", "test"):
            ids = read_split(root, split)
            ordered = stable_order(ids, source_name, seed)
            selected = ordered[:count_for_fraction(len(ordered), fraction)] if split == "train" else ordered
            aliases = [link_sequence(view, root, source_name, sid) for sid in selected]
            split_aliases[split].extend(aliases)
            source_meta["splits"][split] = {"available": len(ids), "selected": len(selected)}
        meta_sources.append(source_meta)

    for split, aliases in split_aliases.items():
        (view / "splits" / f"{split}_sequences.txt").write_text("\n".join(aliases) + ("\n" if aliases else ""))

    meta = {
        "schema_version": "1.0.0",
        "view_name": view_name,
        "seed": seed,
        "training_sequence_fraction": fraction,
        "selection_unit": "sequence",
        "selection_rule": "stable SHA256 ordering; prefix fraction per source",
        "sources": meta_sources,
        "totals": {k: len(v) for k, v in split_aliases.items()},
    }
    (view / "view.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Create deterministic symlink-only dataset views for modular training.")
    ap.add_argument("--isaac-root", default="../data/refined_isaac_0908")
    ap.add_argument("--synthetic-root", default="../data/synth_large_0906")
    ap.add_argument("--output-root", default="../data/modular_views")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cheap-fraction", type=float, default=0.15)
    ap.add_argument("--baseline-fraction", type=float, default=0.50)
    args = ap.parse_args()

    isaac = Path(args.isaac_root)
    synth = Path(args.synthetic_root)
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)

    recipes = [
        ("isaac_15_seed42", [("isaac", isaac)], args.cheap_fraction),
        ("isaac_50_seed42", [("isaac", isaac)], args.baseline_fraction),
        ("isaac_100_seed42", [("isaac", isaac)], 1.0),
        ("combined_15_seed42", [("isaac", isaac), ("synthetic", synth)], args.cheap_fraction),
        ("combined_50_seed42", [("isaac", isaac), ("synthetic", synth)], args.baseline_fraction),
        ("combined_100_seed42", [("isaac", isaac), ("synthetic", synth)], 1.0),
    ]
    for name, sources, fraction in recipes:
        meta = make_view(out, name, sources, fraction, args.seed)
        print(f"{name}: train={meta['totals']['train']} val={meta['totals']['validation']} test={meta['totals']['test']}")


if __name__ == "__main__":
    main()
