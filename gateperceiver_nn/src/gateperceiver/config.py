from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import os
import yaml


def deep_merge(base: dict, overlay: dict) -> dict:
    out = deepcopy(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return _expand_env(cfg)


def save_yaml(obj: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _expand_env(x):
    if isinstance(x, str):
        return os.path.expandvars(x)
    if isinstance(x, list):
        return [_expand_env(v) for v in x]
    if isinstance(x, dict):
        return {k: _expand_env(v) for k, v in x.items()}
    return x
