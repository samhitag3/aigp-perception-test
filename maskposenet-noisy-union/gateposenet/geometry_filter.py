"""Runtime geometry sanity filtering for MaskPoseNet-MG predictions.

The network's presence score says whether a query believes a gate exists.  It
is *not* a guarantee that the query's four predicted corners are geometrically
usable.  In particular, a query can occasionally have presence ~= 1 while its
quad spans a large part of the image.

This module rejects those outputs using only information available at runtime:

* the input UNION gate mask (all non-zero pixels are gate foreground),
* the query's predicted instance mask,
* the query's predicted four corners, and
* the query's predicted center.

No ground truth is needed at inference time.  The checks are deliberately
loose enough to keep partial / cut-off gates while rejecting quads whose edges
have little support in the observed mask.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class GeometryFilterConfig:
    enabled: bool = True

    # Distance, in native input pixels, within which a predicted edge/corner
    # sample is considered supported by the observed union mask.
    support_radius_px: float = 10.0

    # Main anti-giant-quad check: at least this fraction of in-frame samples
    # along the 4 predicted edges must lie near observed gate foreground.
    min_edge_support: float = 0.40

    # At least this fraction of predicted corners that land inside the image
    # must be near observed gate foreground.  Only applied when >=1 corner is
    # in frame; cut-off gates with outside corners are handled by edge support.
    min_corner_support: float = 0.50

    # Predicted instance mask must overlap the observed UNION mask.  This is
    # precision, not IoU, so other gates in the union do not penalize a query.
    min_mask_input_precision: float = 0.12
    min_pred_mask_pixels: int = 10

    # Predicted center head should broadly agree with the quad center.
    max_center_error_frac_diag: float = 0.18

    # Loose guard against absurdly large numerical quads.  Values > 1 allow
    # valid close/cut-off gates to extend beyond the image.
    max_quad_area_frac: float = 2.0
    min_quad_area_px: float = 16.0

    # Sampling density for edge-support test.
    edge_samples_per_side: int = 24
    min_inframe_edge_samples: int = 8

    # Suppress duplicate query slots that predict essentially the same gate.
    duplicate_mask_iou: float = 0.55


def config_from_mapping(mapping: dict[str, Any] | None) -> GeometryFilterConfig:
    """Build config from a YAML mapping while ignoring unknown keys."""
    if not mapping:
        return GeometryFilterConfig()
    allowed = set(GeometryFilterConfig.__dataclass_fields__)
    vals = {k: v for k, v in mapping.items() if k in allowed}
    return GeometryFilterConfig(**vals)


def _distance_to_foreground(union_mask: np.ndarray) -> np.ndarray:
    """Per-pixel distance to nearest foreground pixel."""
    fg = np.asarray(union_mask, dtype=bool)
    # distanceTransform measures non-zero pixels to the nearest zero pixel.
    # Make gate foreground zero and background one.
    background = (~fg).astype(np.uint8)
    return cv2.distanceTransform(background, cv2.DIST_L2, 3)


def _sample_edge_points(corners_px: np.ndarray, samples_per_side: int) -> np.ndarray:
    pts = []
    n = max(2, int(samples_per_side))
    # endpoint=False avoids counting each corner twice.
    ts = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
    for i in range(4):
        a = corners_px[i]
        b = corners_px[(i + 1) % 4]
        pts.append(a[None, :] * (1.0 - ts[:, None]) + b[None, :] * ts[:, None])
    return np.concatenate(pts, axis=0)


def _support_fraction(points_px: np.ndarray, dist_to_fg: np.ndarray,
                      radius_px: float) -> tuple[float, int]:
    h, w = dist_to_fg.shape
    p = np.asarray(points_px, dtype=np.float32)
    finite = np.isfinite(p).all(axis=1)
    inside = finite & (p[:, 0] >= 0) & (p[:, 0] <= w - 1) & \
        (p[:, 1] >= 0) & (p[:, 1] <= h - 1)
    p = p[inside]
    if len(p) == 0:
        return 0.0, 0
    xi = np.clip(np.rint(p[:, 0]).astype(np.int32), 0, w - 1)
    yi = np.clip(np.rint(p[:, 1]).astype(np.int32), 0, h - 1)
    supported = dist_to_fg[yi, xi] <= float(radius_px)
    return float(np.mean(supported)), int(len(p))


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return float(inter / union) if union else 0.0


def _single_query_geometry(
    *,
    corners_px: np.ndarray,
    center_px: np.ndarray,
    pred_mask: np.ndarray,
    union_mask: np.ndarray,
    dist_to_fg: np.ndarray,
    cfg: GeometryFilterConfig,
) -> tuple[bool, str, dict[str, float]]:
    """Return (keep, rejection_reason, diagnostics)."""
    h, w = union_mask.shape
    diag = float(np.hypot(w, h))
    corners = np.asarray(corners_px, dtype=np.float32).reshape(4, 2)
    center = np.asarray(center_px, dtype=np.float32).reshape(2)
    pm = np.asarray(pred_mask, dtype=bool)

    d: dict[str, float] = {}

    if not np.isfinite(corners).all() or not np.isfinite(center).all():
        return False, "nonfinite_geometry", d

    contour = np.rint(corners).astype(np.int32).reshape(-1, 1, 2)
    area = float(abs(cv2.contourArea(corners.astype(np.float32))))
    d["quad_area_frac"] = area / max(float(w * h), 1.0)
    if area < float(cfg.min_quad_area_px):
        return False, "quad_too_small", d
    if d["quad_area_frac"] > float(cfg.max_quad_area_frac):
        return False, "quad_too_large", d

    # D4-equivalent corner orderings are cyclic/reversed, so a physically
    # valid projected square remains a convex perimeter.  Self-crossing or
    # severely disordered predictions are not usable keypoint geometry.
    if not cv2.isContourConvex(contour):
        return False, "quad_nonconvex", d

    quad_center = corners.mean(axis=0)
    center_err = float(np.linalg.norm(quad_center - center))
    d["center_error_frac_diag"] = center_err / max(diag, 1.0)
    if d["center_error_frac_diag"] > float(cfg.max_center_error_frac_diag):
        return False, "center_quad_disagree", d

    pred_pixels = int(pm.sum())
    d["pred_mask_pixels"] = float(pred_pixels)
    if pred_pixels < int(cfg.min_pred_mask_pixels):
        return False, "pred_mask_too_small", d

    inter = int(np.logical_and(pm, union_mask).sum())
    mask_precision = float(inter / max(pred_pixels, 1))
    d["mask_input_precision"] = mask_precision
    if mask_precision < float(cfg.min_mask_input_precision):
        return False, "pred_mask_not_supported", d

    edge_pts = _sample_edge_points(corners, cfg.edge_samples_per_side)
    edge_support, n_edge = _support_fraction(
        edge_pts, dist_to_fg, cfg.support_radius_px)
    d["edge_support"] = edge_support
    d["inframe_edge_samples"] = float(n_edge)
    if n_edge < int(cfg.min_inframe_edge_samples):
        return False, "too_little_visible_edge", d
    if edge_support < float(cfg.min_edge_support):
        return False, "edge_not_supported", d

    corner_support, n_corners = _support_fraction(
        corners, dist_to_fg, cfg.support_radius_px)
    d["corner_support"] = corner_support
    d["inframe_corners"] = float(n_corners)
    if n_corners > 0 and corner_support < float(cfg.min_corner_support):
        return False, "corners_not_supported", d

    return True, "ok", d


def filter_queries(
    *,
    presence: np.ndarray,
    corners_uv: np.ndarray,
    center_uv: np.ndarray,
    pred_masks: np.ndarray,
    input_union_mask: np.ndarray,
    presence_threshold: float,
    cfg: GeometryFilterConfig | None = None,
) -> tuple[list[int], dict[int, dict[str, Any]], Counter]:
    """Filter MaskPoseNet query slots using runtime-observable geometry.

    Args:
        presence: (Q,) probabilities.
        corners_uv: (Q,4,2) normalized coordinates.
        center_uv: (Q,2) normalized coordinates.
        pred_masks: (Q,H,W) binary predicted instance masks at native size.
        input_union_mask: (H,W) binary observed gate mask supplied to model.
        presence_threshold: first-stage confidence threshold.
        cfg: sanity-filter configuration.

    Returns:
        kept query indices, per-query diagnostics, rejection counts.
    """
    cfg = cfg or GeometryFilterConfig()
    pres = np.asarray(presence, dtype=np.float32).reshape(-1)
    corners_uv = np.asarray(corners_uv, dtype=np.float32)
    center_uv = np.asarray(center_uv, dtype=np.float32)
    masks = np.asarray(pred_masks, dtype=bool)
    union = np.asarray(input_union_mask, dtype=bool)
    h, w = union.shape

    diagnostics: dict[int, dict[str, Any]] = {}
    reasons: Counter = Counter()

    candidates = [int(q) for q in np.where(pres >= float(presence_threshold))[0]]
    if not cfg.enabled:
        return candidates, diagnostics, reasons

    # If the detector supplied no gate pixels, no query has visual support.
    if not union.any():
        for q in candidates:
            diagnostics[q] = {"accepted": False, "reason": "empty_input_mask"}
            reasons["empty_input_mask"] += 1
        return [], diagnostics, reasons

    dist_to_fg = _distance_to_foreground(union)
    scale = np.asarray([w, h], dtype=np.float32)

    geometrically_valid: list[int] = []
    for q in candidates:
        ok, reason, d = _single_query_geometry(
            corners_px=corners_uv[q] * scale,
            center_px=center_uv[q] * scale,
            pred_mask=masks[q],
            union_mask=union,
            dist_to_fg=dist_to_fg,
            cfg=cfg,
        )
        diagnostics[q] = {"accepted": bool(ok), "reason": reason, **d}
        if ok:
            geometrically_valid.append(q)
        else:
            reasons[reason] += 1

    # Duplicate suppression: high-confidence query wins when two surviving
    # query masks substantially describe the same observed gate.
    selected: list[int] = []
    for q in sorted(geometrically_valid, key=lambda i: float(pres[i]), reverse=True):
        duplicate_of = None
        for k in selected:
            iou = _mask_iou(masks[q], masks[k])
            if iou >= float(cfg.duplicate_mask_iou):
                duplicate_of = k
                diagnostics[q]["duplicate_iou"] = float(iou)
                break
        if duplicate_of is None:
            selected.append(q)
        else:
            diagnostics[q]["accepted"] = False
            diagnostics[q]["reason"] = "duplicate_query"
            diagnostics[q]["duplicate_of"] = int(duplicate_of)
            reasons["duplicate_query"] += 1

    # Keep stable rendering/order by query index; duplicate winner selection
    # above still used confidence.
    selected.sort()
    return selected, diagnostics, reasons


def config_as_dict(cfg: GeometryFilterConfig) -> dict[str, Any]:
    return asdict(cfg)
