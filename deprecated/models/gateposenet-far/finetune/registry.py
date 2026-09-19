"""Backend + checkpoint registry for the fine-tuning package."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

BACKENDS = ["gatenet", "yolo26", "yoloe"]

# Ultralytics YOLO families. `project` is the output dir (the user created these
# empty dirs: ./yolo26 and ./yolo26e) where runs + cached weights live.
YOLO_FAMILIES = {
    "yolo26": {
        "loader": "YOLO",            # ultralytics.YOLO
        "variants": ["n", "s", "m", "l", "x"],
        "default_variant": "s",
        "ckpt": "yolo26{v}-seg.pt",  # instance-seg checkpoints
        "project": "yolo26",
        "trainer": None,
    },
    "yoloe": {
        "loader": "YOLOE",           # ultralytics.YOLOE
        "variants": ["26n", "26s", "26m", "26l", "26x",
                     "11s", "11m", "11l", "v8s", "v8m", "v8l"],
        "default_variant": "26s",
        "ckpt": "yoloe-{v}-seg.pt",
        "project": "yolo26e",
        "trainer": "YOLOEPESegTrainer",
    },
}


def resolve(model: str, variant: str | None = None) -> dict:
    """Return a resolved spec for a model name.

    For YOLO families returns {family, loader, ckpt, project, trainer, variant}.
    For gatenet returns {family: 'gatenet'}.
    """
    if model == "gatenet":
        return {"family": "gatenet"}
    if model not in YOLO_FAMILIES:
        raise ValueError(f"unknown model '{model}'. Choices: {BACKENDS}")

    fam = YOLO_FAMILIES[model]
    v = variant or fam["default_variant"]
    if v not in fam["variants"]:
        raise ValueError(f"{model} variant '{v}' invalid. Choices: {fam['variants']}")

    project_dir = REPO_ROOT / fam["project"]
    # prefer a user-provided weight in the project dir, else the named checkpoint
    ckpt_name = fam["ckpt"].format(v=v)
    local = project_dir / ckpt_name
    weights = str(local) if local.exists() else ckpt_name

    return {
        "family": model,
        "loader": fam["loader"],
        "weights": weights,
        "ckpt_name": ckpt_name,
        "project": str(project_dir),
        "trainer": fam["trainer"],
        "variant": v,
    }
