"""Tests for the fine-tuning surgery (load / freeze / reset heads / LR groups)."""

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatenet.finetune import (configure_finetune, load_pretrained, param_groups,
                              reset_output_heads, set_trainable)
from gatenet.model import build_gatenet


def _model():
    return build_gatenet({"width_factor": 2})


def test_load_pretrained_roundtrip(tmp_path=None):
    tmp = Path(tmp_path or tempfile.mkdtemp())
    base = _model()
    ckpt = tmp / "base.pt"
    torch.save({"model": base.state_dict()}, ckpt)

    other = _model()
    load_pretrained(other, str(ckpt), strict=True)
    for (n1, p1), (n2, p2) in zip(base.named_parameters(), other.named_parameters()):
        assert torch.allclose(p1, p2), n1


def test_freeze_encoder():
    m = _model()
    set_trainable(m, freeze_encoder=True)
    enc = [p.requires_grad for n, p in m.named_parameters() if n.startswith("down1")]
    dec = [p.requires_grad for n, p in m.named_parameters() if n.startswith("up4")]
    heads = [p.requires_grad for n, p in m.named_parameters() if n.startswith("outc0")]
    assert not any(enc), "encoder should be frozen"
    assert all(dec), "decoder should be trainable"
    assert all(heads), "heads should be trainable"


def test_reset_heads_changes_weights():
    m = _model()
    before = m.outc0.conv.weight.clone()
    torch.manual_seed(123)
    reset_output_heads(m)
    after = m.outc0.conv.weight
    assert not torch.allclose(before, after), "heads should be re-initialised"


def test_param_groups_discriminative():
    m = _model()
    set_trainable(m, freeze_encoder=False)
    groups = param_groups(m, base_lr=1e-3, encoder_lr_mult=0.1)
    lrs = sorted(g["lr"] for g in groups)
    assert len(groups) == 2
    assert abs(lrs[0] - 1e-4) < 1e-9   # encoder = base * 0.1
    assert abs(lrs[1] - 1e-3) < 1e-9   # decoder = base


def test_configure_finetune_freeze(tmp_path=None):
    tmp = Path(tmp_path or tempfile.mkdtemp())
    ckpt = tmp / "base.pt"
    torch.save({"model": _model().state_dict()}, ckpt)
    m = _model()
    configure_finetune(m, str(ckpt), freeze_encoder=True, reset_heads=True)
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in m.parameters())
    assert 0 < n_train < n_total


if __name__ == "__main__":
    test_load_pretrained_roundtrip()
    test_freeze_encoder()
    test_reset_heads_changes_weights()
    test_param_groups_discriminative()
    test_configure_finetune_freeze()
    print("all finetune tests passed")
