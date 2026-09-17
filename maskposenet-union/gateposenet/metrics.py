"""Validation metrics for GatePoseNet, bucketed by range.

All computed on VISIBLE frames unless stated; blind-frame (memory) position
error is reported separately.
"""

from __future__ import annotations

import numpy as np
import torch

from .losses import CORNER_PERMS, gate_symmetry_group


@torch.no_grad()
def pose_metrics(pred: dict, tgt: dict, img_wh: tuple[int, int],
                 symmetry_aware: bool = True) -> dict:
    """Aggregate error metrics for one batch of (B, T, ...) outputs.

    With ``symmetry_aware`` (default) rotation / corner / axis errors are
    computed modulo the gate's D4 symmetry (matching the training loss).
    """
    W, H = img_wh
    frame_ok = tgt.get("frame_ok")
    fok = (frame_ok > 0.5) if frame_ok is not None \
        else torch.ones_like(tgt["visible"]).bool()
    vis = (tgt["visible"] > 0.5) & fok
    pose_ok = (tgt["pose_ok"] > 0.5) & fok
    out: dict[str, list] = {}

    def add(key, val_tensor, mask):
        if mask.any():
            out.setdefault(key, []).append(
                val_tensor[mask].detach().float().cpu().numpy())

    # position (m) and relative-to-range
    pos_err = (pred["position"] - tgt["position"]).norm(dim=-1)
    rng = tgt["position"].norm(dim=-1).clamp_min(1e-6)
    add("pos_err_m", pos_err, vis & pose_ok)
    add("pos_err_rel", pos_err / rng, vis & pose_ok)
    add("pos_err_m_blind", pos_err, (~vis) & pose_ok)

    # geodesic rotation error (deg), min over the symmetry group
    R_t = tgt["R"]
    if symmetry_aware:
        G = gate_symmetry_group().to(R_t.device)
        Rg = torch.einsum("btij,gjk->btgik", R_t, G)
        Rt = pred["R"][:, :, None].transpose(-1, -2) @ Rg
        tr = Rt.diagonal(dim1=-2, dim2=-1).sum(-1)
        ang = torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1, 1)))
        ang = ang.min(dim=-1).values
    else:
        Rt = pred["R"].transpose(-1, -2) @ R_t
        tr = Rt.diagonal(dim1=-2, dim2=-1).sum(-1)
        ang = torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1, 1)))
    add("rot_err_deg", ang, vis & pose_ok)

    # fly-through AXIS error (deg): symmetry-invariant up to sign.
    n_p = -pred["R"][..., :, 2]
    n_t = -R_t[..., :, 2]
    cosang = (n_p * n_t).sum(-1).abs().clamp(-1, 1) if symmetry_aware else \
        (n_p * n_t).sum(-1).clamp(-1, 1)
    add("normal_err_deg", torch.rad2deg(torch.arccos(cosang)), vis & pose_ok)

    # 2-D errors in PIXELS of each frame's TRUE image size (dataset supplies
    # img_wh per frame; config native size is the fallback).
    if "img_wh" in tgt:
        scale = tgt["img_wh"]                             # (B,T,2)
    else:
        scale = torch.tensor([W, H], dtype=torch.float32,
                             device=pred["center_uv"].device)
    c_err = ((pred["center_uv"] - tgt["center_uv"]) * scale).norm(dim=-1)
    add("center_err_px", c_err, (tgt["center_ok"] > 0.5) & fok)
    if symmetry_aware:
        perms = CORNER_PERMS.to(R_t.device)
        tc = tgt["corners_uv"][:, :, perms]               # (B,T,8,4,2)
        s8 = scale[:, :, None, None, :] if scale.dim() == 3 else scale
        k_err_all = ((pred["corners_uv"][:, :, None] - tc) * s8).norm(dim=-1)
        k_err = k_err_all.mean(-1).min(dim=-1).values     # (B,T)
    else:
        s4 = scale[:, :, None, :] if scale.dim() == 3 else scale
        k_err = ((pred["corners_uv"] - tgt["corners_uv"]) * s4)\
            .norm(dim=-1).mean(-1)
    corner_valid = vis & (tgt["corner_ok"].sum(-1) > 3.5)
    add("corner_err_px", k_err, corner_valid)

    # visibility classification accuracy (padded frames excluded)
    v_acc = ((pred["visible_logit"] > 0) == (tgt["visible"] > 0.5)).float()
    out.setdefault("vis_acc", []).append(
        v_acc[fok].flatten().cpu().numpy())

    # range-bucketed relative position error
    for lo, hi in [(0, 3), (3, 6), (6, 10), (10, 25)]:
        m = vis & pose_ok & (rng >= lo) & (rng < hi)
        add(f"pos_rel_{lo}-{hi}m", pos_err / rng, m)
    return out


def reduce_metrics(acc: dict) -> dict:
    """Concatenate batch lists -> {name: (mean, median)} floats."""
    out = {}
    for k, chunks in acc.items():
        v = np.concatenate(chunks) if chunks else np.array([np.nan])
        out[k] = float(np.mean(v))
        out[k + "_med"] = float(np.median(v))
    return out


def merge_metric_acc(dst: dict, src: dict) -> None:
    for k, v in src.items():
        dst.setdefault(k, []).extend(v)
