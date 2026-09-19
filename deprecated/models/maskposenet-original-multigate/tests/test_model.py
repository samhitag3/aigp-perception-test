"""Smoke tests for the GateNet model, loss and metrics.

Run:  python -m pytest tests/ -q     (or)     python tests/test_model.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatenet.losses import DeepSupervisionLoss
from gatenet.metrics import seg_metrics
from gatenet.model import GateNetInference, build_gatenet


def _model(width_factor=2, size=64):
    cfg = {"in_channels": 3, "width_factor": width_factor, "deep_supervision": True}
    return build_gatenet(cfg)


def test_forward_shapes():
    model = _model()
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert len(out) == 5
    # highest-res output matches input HxW; each subsequent is halved
    assert out[0].shape == (2, 1, 64, 64)
    assert out[1].shape == (2, 1, 32, 32)
    assert out[2].shape == (2, 1, 16, 16)
    assert out[3].shape == (2, 1, 8, 8)
    assert out[4].shape == (2, 1, 4, 4)


def test_backward():
    model = _model()
    crit = DeepSupervisionLoss()
    x = torch.randn(2, 3, 64, 64)
    y = (torch.rand(2, 1, 64, 64) > 0.5).float()
    out = model(x)
    loss, logs = crit(out, y)
    loss.backward()
    assert torch.isfinite(loss)
    assert "loss/total" in logs
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_metrics_perfect():
    logits = torch.full((1, 1, 8, 8), 10.0)   # all "gate"
    target = torch.ones(1, 1, 8, 8)
    m = seg_metrics(logits, target)
    assert m["iou"] > 0.99 and m["dice"] > 0.99


def test_inference_wrapper():
    model = _model()
    wrap = GateNetInference(model, out_size=32, threshold=0.5).eval()
    with torch.no_grad():
        out = wrap(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 1, 32, 32)
    assert set(torch.unique(out).tolist()).issubset({0.0, 1.0})


def test_non_square_input():
    model = _model()
    out = model(torch.randn(1, 3, 96, 128))
    assert out[0].shape[-2:] == (96, 128)


if __name__ == "__main__":
    test_forward_shapes()
    test_backward()
    test_metrics_perfect()
    test_inference_wrapper()
    test_non_square_input()
    print("all model tests passed")
