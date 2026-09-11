from __future__ import annotations
import argparse
import json
from pathlib import Path

from gateperception.data.canonical import CanonicalIndex
from gateperception.utils.config import load_yaml


def sources_from_dataset(d: dict, validation: bool = False):
    key = "validation_sources" if validation and d.get("validation_sources") else "sources"
    if d.get(key):
        out = []
        for i, src in enumerate(d[key]):
            if isinstance(src, str):
                out.append((f"source_{i}", src))
            else:
                out.append((str(src.get("name", f"source_{i}")), str(src["root"])))
        return out
    return [(str(d.get("name", "dataset")), str(d["root"]))]


def main():
    ap = argparse.ArgumentParser(description="Show the deterministic sequence subset selected by a training config.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", default=None, help="Optional JSON path to save the selection.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    d = cfg["dataset"]
    seed = int(d.get("selection_seed", cfg.get("seed", 42)))
    train_fraction = float(d.get("train_fraction", 1.0))
    val_fraction = float(d.get("val_fraction", 1.0))
    report = {
        "config": args.config,
        "seed": int(cfg.get("seed", 42)),
        "selection_seed": seed,
        "train_fraction": train_fraction,
        "validation_fraction": val_fraction,
        "train": {},
        "validation": {},
    }
    for name, root in sources_from_dataset(d, validation=False):
        idx = CanonicalIndex(
            root,
            d.get("train_splits", ["train"]),
            max_sequences=d.get("max_train_sequences"),
            sequence_fraction=train_fraction,
            selection_seed=seed,
        )
        report["train"][name] = {"root": root, "count": len(idx.sequence_ids), "sequence_ids": idx.sequence_ids}
    for name, root in sources_from_dataset(d, validation=True):
        idx = CanonicalIndex(
            root,
            d.get("val_splits", ["validation"]),
            max_sequences=d.get("max_val_sequences"),
            sequence_fraction=val_fraction,
            selection_seed=seed,
        )
        report["validation"][name] = {"root": root, "count": len(idx.sequence_ids), "sequence_ids": idx.sequence_ids}

    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        p = Path(args.output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
