"""Fine-tuning helpers for GateNet.

Typical use: take a base GateNet trained on synthetic + a large real set, then
adapt it to a *new* gate type or venue (e.g. orange -> MAVLab, or a new hall's
lighting) using a small labeled set. This mirrors how MonoRace / SkyDreamer
deploy GateNet across gate types.

Three knobs cover the common regimes:
  * ``freeze_encoder``   : train only the decoder + output heads (fast, small
    data — the encoder's low-level edge/colour features usually transfer).
  * ``reset_heads``      : re-initialise the 1x1 output heads (when the new
    domain looks very different and you want the heads to relearn).
  * ``encoder_lr_mult``  : discriminative LR — encoder learns slower than the
    decoder (e.g. 0.1) for a gentle full fine-tune.

See scripts/finetune.py for the CLI.
"""

from __future__ import annotations

import torch

from .model import GateNet, OutConv, xavier_init


# encoder = feature extractor; decoder+heads = task-specific
_ENCODER_PREFIXES = ("inc", "down1", "down2", "down3", "down4")
_HEAD_PREFIXES = ("outc0", "outc1", "outc2", "outc3", "outc4")


def load_pretrained(model: GateNet, ckpt_path: str, map_location="cpu",
                    strict: bool = False) -> dict:
    """Load pretrained weights into ``model`` (weights only, not optimizer/epoch).

    Returns the {missing, unexpected} key report. ``strict=False`` tolerates
    architecture tweaks (e.g. width changes you intend to partially load).
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)
    state = ckpt.get("model", ckpt)
    result = model.load_state_dict(state, strict=strict)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    if missing:
        print(f"[finetune] {len(missing)} missing keys (left at init), e.g. {missing[:3]}")
    if unexpected:
        print(f"[finetune] {len(unexpected)} unexpected keys ignored, e.g. {unexpected[:3]}")
    return {"missing": missing, "unexpected": unexpected}


def reset_output_heads(model: GateNet) -> None:
    """Re-initialise the 1x1 output heads with Xavier-uniform."""
    for name, m in model.named_modules():
        if isinstance(m, OutConv):
            xavier_init(m)
    print("[finetune] output heads re-initialised")


def set_trainable(model: GateNet, freeze_encoder: bool = False,
                  freeze_bn_stats: bool = False) -> None:
    """Freeze/unfreeze parameter groups.

    Args:
        freeze_encoder: if True, encoder params (inc, down1..down4) are frozen.
        freeze_bn_stats: if True, encoder BatchNorm layers are put in eval mode
            so their running stats are NOT updated (recommended with tiny data).
    """
    for name, p in model.named_parameters():
        is_encoder = name.split(".")[0] in _ENCODER_PREFIXES
        p.requires_grad = not (freeze_encoder and is_encoder)

    if freeze_encoder and freeze_bn_stats:
        for name, m in model.named_modules():
            if name.split(".")[0] in _ENCODER_PREFIXES and isinstance(m, torch.nn.BatchNorm2d):
                m.eval()

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[finetune] trainable params: {n_train/1e6:.3f}M / {n_total/1e6:.3f}M "
          f"({100*n_train/n_total:.1f}%)")


def param_groups(model: GateNet, base_lr: float, encoder_lr_mult: float = 1.0,
                 weight_decay: float = 0.01):
    """Discriminative-LR parameter groups: encoder gets base_lr*encoder_lr_mult."""
    enc, dec = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (enc if name.split(".")[0] in _ENCODER_PREFIXES else dec).append(p)
    groups = []
    if dec:
        groups.append({"params": dec, "lr": base_lr, "weight_decay": weight_decay})
    if enc:
        groups.append({"params": enc, "lr": base_lr * encoder_lr_mult,
                       "weight_decay": weight_decay})
    return groups


def configure_finetune(model: GateNet, ckpt_path: str, *, freeze_encoder: bool = False,
                       reset_heads: bool = False, freeze_bn_stats: bool = False,
                       map_location="cpu") -> GateNet:
    """One-call setup: load weights, optionally reset heads, set trainability."""
    load_pretrained(model, ckpt_path, map_location=map_location, strict=False)
    if reset_heads:
        reset_output_heads(model)
    set_trainable(model, freeze_encoder=freeze_encoder, freeze_bn_stats=freeze_bn_stats)
    return model
