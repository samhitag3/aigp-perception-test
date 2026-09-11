"""Ultralytics YOLO26 / YOLOE fine-tuning backend for gate segmentation.

The autolabeler dataset is already in YOLO-seg format (polygon labels +
data.yaml), so fine-tuning = loading a pretrained seg checkpoint and calling
``model.train(data=..., ...)``. Outputs go to the per-family project dir
(./yolo26 for YOLO26, ./yolo26e for YOLOE).

Requires the optional ``ultralytics`` dependency:  uv sync --extra yolo
"""

from __future__ import annotations

from pathlib import Path

from ..registry import resolve
from .base import Backend


class YoloBackend(Backend):
    def __init__(self, model: str):
        self.name = model  # 'yolo26' or 'yoloe'

    def _import_ultralytics(self, loader: str):
        try:
            import ultralytics  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "ultralytics is not installed. Run:  uv sync --extra yolo\n"
                "(YOLO26 / YOLOE require a recent ultralytics release.)"
            ) from e
        from ultralytics import YOLO, YOLOE  # type: ignore
        return YOLO if loader == "YOLO" else YOLOE

    def finetune(self, args) -> dict:
        spec = resolve(self.name, args.variant)
        Loader = self._import_ultralytics(spec["loader"])

        # Absolute project dir so Ultralytics saves under ./yolo26 (or ./yolo26e),
        # not under its default runs/segment/. spec["project"] is already absolute.
        project = str(Path(args.output).resolve()) if args.output else spec["project"]
        Path(project).mkdir(parents=True, exist_ok=True)
        run_name = args.name or f"{spec['family']}-{spec['variant']}-finetune"

        weights = args.weights or spec["weights"]
        print(f"[{self.name}] loading {weights}  (variant {spec['variant']})")
        model = Loader(weights)

        train_kw = dict(
            data=args.data,
            epochs=args.epochs or 100,
            imgsz=args.imgsz or 640,
            batch=args.batch or 16,
            device=args.device if args.device is not None else 0,
            project=project,
            name=run_name,
            patience=args.patience,
            seed=args.seed,
            pretrained=True,            # fine-tune from the loaded weights
        )

        # YOLOE: optionally restrict to the dataset's class set and use its
        # linear-probing segmentation trainer.
        if spec["trainer"]:
            try:
                from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer  # type: ignore
                train_kw["trainer"] = YOLOEPESegTrainer
            except Exception as e:  # pragma: no cover
                print(f"[yoloe] could not import YOLOEPESegTrainer ({e}); using default trainer")
            if args.classes and hasattr(model, "set_classes"):
                try:
                    model.set_classes(list(args.classes))
                except Exception:
                    pass

        print(f"[{self.name}] train: {train_kw}")
        results = model.train(**train_kw)

        save_dir = getattr(results, "save_dir", str(Path(project) / run_name))
        return {"backend": self.name, "variant": spec["variant"],
                "save_dir": str(save_dir), "weights": weights}
