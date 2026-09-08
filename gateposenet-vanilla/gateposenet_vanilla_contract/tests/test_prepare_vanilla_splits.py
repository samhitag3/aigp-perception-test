from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _ids(path: Path):
    return [x.strip() for x in path.read_text().splitlines() if x.strip()]


def test_prepare_splits_is_deterministic_nested_and_preserves_official(tmp_path):
    root = tmp_path / "data"
    seqs = root / "sequences"
    for i in range(20):
        (seqs / f"seq_{i:03d}").mkdir(parents=True)
    splits = root / "splits"
    splits.mkdir(parents=True)
    train = [f"seq_{i:03d}" for i in range(16)]
    val = ["seq_016", "seq_017"]
    test = ["seq_018", "seq_019"]
    (splits / "train_sequences.txt").write_text("\n".join(train) + "\n")
    (splits / "validation_sequences.txt").write_text("\n".join(val) + "\n")
    (splits / "test_sequences.txt").write_text("\n".join(test) + "\n")

    script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_vanilla_splits.py"
    cmd = [sys.executable, str(script), "--dataset-root", str(root), "--seed", "42"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    cheap1 = _ids(splits / "vanilla_seed42" / "train_cheap_sequences.txt")
    base1 = _ids(splits / "vanilla_seed42" / "train_baseline_sequences.txt")
    tune1 = _ids(splits / "vanilla_seed42" / "train_tune_sequences.txt")
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert cheap1 == _ids(splits / "vanilla_seed42" / "train_cheap_sequences.txt")
    assert base1 == _ids(splits / "vanilla_seed42" / "train_baseline_sequences.txt")
    assert set(cheap1).issubset(set(base1))
    assert base1 == tune1
    assert _ids(splits / "train_sequences.txt") == train
    info = json.loads((splits / "vanilla_seed42" / "manifest_info.json").read_text())
    assert info["seed"] == 42
    assert info["nested"] is True
