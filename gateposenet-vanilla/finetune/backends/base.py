"""Backend interface for fine-tuning."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Backend(ABC):
    name: str = "base"

    @abstractmethod
    def finetune(self, args) -> dict:
        """Run fine-tuning. Returns a small result dict (best metric, out dir)."""
        raise NotImplementedError


def get_backend(model: str) -> Backend:
    """Factory: map a model name to its backend instance."""
    if model == "gatenet":
        from .gatenet_backend import GateNetBackend
        return GateNetBackend()
    if model in ("yolo26", "yoloe"):
        from .yolo_backend import YoloBackend
        return YoloBackend(model)
    raise ValueError(f"unknown model '{model}'")
