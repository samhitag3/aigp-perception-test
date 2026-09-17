"""Standard evaluation/report writer for vanilla GatePoseNetSingle.

Produces the project-wide ``evaluation.json`` plus ``per_sample_metrics.csv``
without changing the model architecture. Unsupported multi-instance/8-keypoint
metrics are explicitly marked unavailable.
"""

from __future__ import annotations

import csv
import json
import math
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .losses import CORNER_PERMS, gate_symmetry_group
from .data_sources import describe_sources


def _finite_or_none(x: Any):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _stats(values: list[float]) -> dict[str, float | None]:
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {
            "mean": None, "median": None, "std": None, "rmse": None,
            "p90": None, "p95": None, "p99": None,
        }
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std()),
        "rmse": float(np.sqrt(np.mean(a * a))),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
    }


def _rate(n: float, d: float) -> float | None:
    return float(n / d) if d > 0 else None


def _safe_json(obj):
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def _sym_rotation_error_deg(Rp: torch.Tensor, Rt: torch.Tensor) -> torch.Tensor:
    G = gate_symmetry_group().to(Rp.device, dtype=Rp.dtype)
    Rg = torch.einsum("nij,gjk->ngik", Rt, G)
    rel = Rp[:, None].transpose(-1, -2) @ Rg
    tr = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    ang = torch.rad2deg(torch.arccos(((tr - 1.0) / 2.0).clamp(-1, 1)))
    return ang.min(dim=-1).values


def _sym_corner_error_px(pred_uv: torch.Tensor, tgt_uv: torch.Tensor,
                         scale_wh: torch.Tensor) -> torch.Tensor:
    perms = CORNER_PERMS.to(pred_uv.device)
    tc = tgt_uv[:, perms]  # (N,8,4,2)
    err = ((pred_uv[:, None] - tc) * scale_wh[:, None, None, :]).norm(dim=-1)
    return err.mean(-1).min(dim=-1).values




def _benchmark_step_latency(model, dataset, device: torch.device, cfg: dict) -> list[float]:
    """Benchmark the actual deployment API: step(image, ego, hidden), batch=1."""
    try:
        sample = dataset[0]
    except Exception:
        return []
    image = sample["image"][-1:].to(device)  # (1,3,H,W)
    ego = sample["ego"][-1:].to(device)      # (1,7)
    warmup = int(cfg.get("evaluation", {}).get("runtime_warmup_steps", 10))
    n = int(cfg.get("evaluation", {}).get("runtime_benchmark_steps", 50))
    h = None
    values = []
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            _, h = model.step(image, ego, h)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for _ in range(max(1, n)):
            if device.type == "cuda":
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                starter.record()
                _, h = model.step(image, ego, h)
                ender.record()
                torch.cuda.synchronize(device)
                values.append(float(starter.elapsed_time(ender)))
            else:
                t0 = time.perf_counter()
                _, h = model.step(image, ego, h)
                values.append((time.perf_counter() - t0) * 1000.0)
    return values


def evaluate_gateposenet_contract(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    cfg: dict,
    split_name: str,
    output_dir: str | Path,
    checkpoint_path: str | Path | None = None,
) -> dict:
    """Evaluate decoded legacy outputs and write canonical experiment artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    dcfg = cfg["data"]
    catastrophic_pos = float(cfg.get("evaluation", {}).get(
        "catastrophic_translation_m", 0.75))
    catastrophic_rot = float(cfg.get("evaluation", {}).get(
        "catastrophic_rotation_deg", 25.0))
    mask_threshold = float(cfg.get("evaluation", {}).get("mask_threshold", 0.5))

    pos_errs: list[float] = []
    rot_errs: list[float] = []
    depth_errs: list[float] = []
    lateral_errs: list[float] = []
    vertical_errs: list[float] = []
    x_errs: list[float] = []
    y_errs: list[float] = []
    z_errs: list[float] = []
    corner_errs: list[float] = []
    norm_corner_errs: list[float] = []
    center_errs: list[float] = []
    seg_ious: list[float] = []
    seg_dices: list[float] = []
    window_latencies_ms: list[float] = []
    per_sample: list[dict[str, Any]] = []

    tp_px = fp_px = fn_px = tn_px = 0.0
    vis_tp = vis_fp = vis_tn = vis_fn = 0
    n_frames = n_pose = n_visible = catastrophic = 0
    pck_counts = {0.01: 0, 0.02: 0, 0.05: 0, 0.10: 0}
    pck_total = 0
    distance_buckets: dict[str, list[float]] = {
        "0_to_2m": [], "2_to_5m": [], "5_to_10m": [], "10m_plus": []}

    use_cuda_timing = device.type == "cuda"
    if use_cuda_timing:
        torch.cuda.reset_peak_memory_stats(device)

    global_sample = 0
    global_window = 0
    with torch.no_grad():
        for batch in loader:
            # Preserve a few CPU values before transfer for metadata-free rows.
            B, T = batch["image"].shape[:2]
            batch_meta = []
            for b in range(B):
                if hasattr(loader.dataset, "window_meta"):
                    meta = dict(loader.dataset.window_meta(global_window + b))
                else:
                    meta = {}
                if "source_name" not in meta:
                    names = getattr(loader.dataset, "source_names", None)
                    if names and len(names) == 1:
                        meta["source_name"] = names[0]
                batch_meta.append(meta)
            global_window += B
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            if use_cuda_timing:
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                starter.record()
                pred = model(batch["image"], batch["ego"])
                ender.record()
                torch.cuda.synchronize(device)
                batch_ms = float(starter.elapsed_time(ender))
            else:
                t0 = time.perf_counter()
                pred = model(batch["image"], batch["ego"])
                batch_ms = (time.perf_counter() - t0) * 1000.0
            window_latencies_ms.extend([batch_ms / max(B, 1)] * B)

            frame_ok = batch.get("frame_ok", torch.ones(B, T, device=device)) > 0.5

            # Evaluate every valid timestep. This mirrors the training metrics.
            for t in range(T):
                valid = frame_ok[:, t]
                if not valid.any():
                    continue
                idx = valid.nonzero(as_tuple=False).flatten()
                n = len(idx)
                n_frames += n

                seg_prob = torch.sigmoid(pred["seg_logit"][idx, t])
                seg_pred = seg_prob >= mask_threshold
                seg_tgt = batch["seg"][idx, t] > 0.5
                dims = (1, 2, 3)
                inter = (seg_pred & seg_tgt).sum(dim=dims).float()
                union = (seg_pred | seg_tgt).sum(dim=dims).float()
                ps = seg_pred.sum(dim=dims).float()
                gs = seg_tgt.sum(dim=dims).float()
                iou = torch.where(union > 0, inter / union,
                                  torch.ones_like(union))
                dice = torch.where(ps + gs > 0, 2 * inter / (ps + gs),
                                   torch.ones_like(inter))
                seg_ious.extend(iou.cpu().tolist())
                seg_dices.extend(dice.cpu().tolist())

                tp_px += float((seg_pred & seg_tgt).sum())
                fp_px += float((seg_pred & ~seg_tgt).sum())
                fn_px += float((~seg_pred & seg_tgt).sum())
                tn_px += float((~seg_pred & ~seg_tgt).sum())

                gt_vis = batch["visible"][idx, t] > 0.5
                pr_vis = pred["visible_logit"][idx, t] > 0
                vis_tp += int((pr_vis & gt_vis).sum())
                vis_fp += int((pr_vis & ~gt_vis).sum())
                vis_tn += int((~pr_vis & ~gt_vis).sum())
                vis_fn += int((~pr_vis & gt_vis).sum())
                n_visible += int(gt_vis.sum())

                pose_ok = batch["pose_ok"][idx, t] > 0.5
                if pose_ok.any():
                    j = pose_ok.nonzero(as_tuple=False).flatten()
                    gi = idx[j]
                    pp = pred["position"][gi, t]
                    gp = batch["position"][gi, t]
                    pe = (pp - gp).norm(dim=-1)
                    re = _sym_rotation_error_deg(pred["R"][gi, t], batch["R"][gi, t])
                    de = (pp[:, 2] - gp[:, 2]).abs()
                    xe = (pp[:, 0] - gp[:, 0]).abs()
                    ye = (pp[:, 1] - gp[:, 1]).abs()
                    ze = de
                    lateral = xe
                    vertical = ye
                    pos_errs.extend(pe.cpu().tolist())
                    rot_errs.extend(re.cpu().tolist())
                    depth_errs.extend(de.cpu().tolist())
                    x_errs.extend(xe.cpu().tolist())
                    y_errs.extend(ye.cpu().tolist())
                    z_errs.extend(ze.cpu().tolist())
                    lateral_errs.extend(lateral.cpu().tolist())
                    vertical_errs.extend(vertical.cpu().tolist())
                    n_pose += len(j)

                    ranges = gp.norm(dim=-1)
                    for e, r in zip(pe.cpu().tolist(), ranges.cpu().tolist()):
                        if r < 2:
                            distance_buckets["0_to_2m"].append(e)
                        elif r < 5:
                            distance_buckets["2_to_5m"].append(e)
                        elif r < 10:
                            distance_buckets["5_to_10m"].append(e)
                        else:
                            distance_buckets["10m_plus"].append(e)

                c_ok = batch["corner_ok"][idx, t].sum(-1) > 3.5
                if c_ok.any():
                    j = c_ok.nonzero(as_tuple=False).flatten()
                    gi = idx[j]
                    scale = batch["img_wh"][gi, t]
                    ce = _sym_corner_error_px(
                        pred["corners_uv"][gi, t], batch["corners_uv"][gi, t], scale)
                    corner_errs.extend(ce.cpu().tolist())
                    diag = torch.sqrt(scale[:, 0] ** 2 + scale[:, 1] ** 2)
                    ne = ce / diag.clamp_min(1.0)
                    norm_corner_errs.extend(ne.cpu().tolist())
                    pck_total += len(ce)
                    for th in pck_counts:
                        pck_counts[th] += int((ne <= th).sum())

                cent_ok = batch["center_ok"][idx, t] > 0.5
                if cent_ok.any():
                    j = cent_ok.nonzero(as_tuple=False).flatten()
                    gi = idx[j]
                    scale = batch["img_wh"][gi, t]
                    ce = ((pred["center_uv"][gi, t] - batch["center_uv"][gi, t])
                          * scale).norm(dim=-1)
                    center_errs.extend(ce.cpu().tolist())

                # Compact row per valid sample; overlapping windows can repeat
                # a frame, so sample_index is the stable evaluation-row key.
                for local_k, bidx in enumerate(idx.cpu().tolist()):
                    meta = batch_meta[bidx] if bidx < len(batch_meta) else {}
                    frame_indices = meta.get("frame_indices") or []
                    frame_index = frame_indices[t] if t < len(frame_indices) else None
                    row = {
                        "sample_index": global_sample,
                        "source": meta.get("source_name"),
                        "sequence_id": meta.get("sequence_id"),
                        "frame_index": frame_index,
                        "window_t": t,
                        "gt_visible": int(bool(gt_vis[local_k].item())),
                        "pred_visible": int(bool(pr_vis[local_k].item())),
                        "union_iou": float(iou[local_k].item()),
                        "union_dice": float(dice[local_k].item()),
                        "gate_distance_m": float(batch["position"][bidx, t].norm().item()),
                        "translation_error_m": None,
                        "rotation_error_deg": None,
                        "depth_error_m": None,
                        "outer_corner_error_px": None,
                        "catastrophic_failure": 0,
                    }
                    if bool(batch["pose_ok"][bidx, t] > 0.5):
                        pe1 = float((pred["position"][bidx, t] - batch["position"][bidx, t]).norm().item())
                        re1 = float(_sym_rotation_error_deg(
                            pred["R"][bidx:bidx+1, t], batch["R"][bidx:bidx+1, t])[0].item())
                        de1 = float(abs(pred["position"][bidx, t, 2].item()
                                        - batch["position"][bidx, t, 2].item()))
                        row.update(translation_error_m=pe1,
                                   rotation_error_deg=re1,
                                   depth_error_m=de1)
                        fail = (pe1 > catastrophic_pos or re1 > catastrophic_rot
                                or (bool(gt_vis[local_k]) and not bool(pr_vis[local_k])))
                        row["catastrophic_failure"] = int(fail)
                        catastrophic += int(fail)
                    if bool(c_ok[local_k]):
                        # Recompute single-row corner error for direct mapping.
                        ce1 = _sym_corner_error_px(
                            pred["corners_uv"][bidx:bidx+1, t],
                            batch["corners_uv"][bidx:bidx+1, t],
                            batch["img_wh"][bidx:bidx+1, t])[0]
                        row["outer_corner_error_px"] = float(ce1.item())
                    per_sample.append(row)
                    global_sample += 1


    by_source_rows: dict[str, list[dict[str, Any]]] = {}
    for row in per_sample:
        src = str(row.get("source") or "unknown")
        by_source_rows.setdefault(src, []).append(row)
    by_source = {}
    for src, rows in by_source_rows.items():
        vals = lambda key: [float(r[key]) for r in rows if r.get(key) is not None]
        cats = [int(r.get("catastrophic_failure", 0)) for r in rows
                if r.get("translation_error_m") is not None]
        by_source[src] = {
            "n_frame_steps": len(rows),
            "union_iou": {
                "mean": _stats(vals("union_iou"))["mean"],
                "median": _stats(vals("union_iou"))["median"],
            },
            "translation_error_m": {
                "mean": _stats(vals("translation_error_m"))["mean"],
                "median": _stats(vals("translation_error_m"))["median"],
                "p95": _stats(vals("translation_error_m"))["p95"],
            },
            "rotation_error_deg": {
                "mean": _stats(vals("rotation_error_deg"))["mean"],
                "median": _stats(vals("rotation_error_deg"))["median"],
                "p95": _stats(vals("rotation_error_deg"))["p95"],
            },
            "catastrophic_failure_rate": _rate(sum(cats), len(cats)),
        }

    precision = _rate(tp_px, tp_px + fp_px)
    recall = _rate(tp_px, tp_px + fn_px)
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall > 0 else None)
    vis_precision = _rate(vis_tp, vis_tp + vis_fp)
    vis_recall = _rate(vis_tp, vis_tp + vis_fn)
    vis_f1 = (2 * vis_precision * vis_recall / (vis_precision + vis_recall)
              if vis_precision is not None and vis_recall is not None
              and vis_precision + vis_recall > 0 else None)

    pos = _stats(pos_errs)
    rot = _stats(rot_errs)
    depth = _stats(depth_errs)
    corners = _stats(corner_errs)
    norm_corners = _stats(norm_corner_errs)
    seg_iou = _stats(seg_ious)
    seg_dice = _stats(seg_dices)
    window_latency = _stats(window_latencies_ms)
    step_latencies_ms = _benchmark_step_latency(model, loader.dataset, device, cfg)
    latency = _stats(step_latencies_ms)

    peak_vram = (float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
                 if use_cuda_timing else None)
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    camera = cfg.get("camera", {})
    report = {
        "schema_version": "1.0.0",
        "run": {
            "run_id": Path(output_dir).name,
            "experiment_name": cfg.get("name", "gateposenet_vanilla_contract"),
            "model_name": "GatePoseNetSingle",
            "model_family": "original_gateposenet_single_target",
            "task_type": "temporal_single_target_gate_pose",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "random_seed": cfg.get("train", {}).get("seed"),
            "tags": ["vanilla", "canonical-data-contract", "single-target"],
        },
        "model": {
            "architecture": "UNCHANGED original GatePoseNetSingle",
            "temporal": True,
            "temporal_window_size": int(dcfg.get("window", 8)),
            "input": {
                "modalities": ["rgb", "ego_motion"],
                "model_image_width": int(dcfg.get("width", 320)),
                "model_image_height": int(dcfg.get("height", 192)),
                "native_image_width": int(dcfg.get("native_width", 640)),
                "native_image_height": int(dcfg.get("native_height", 360)),
                "uses_camera_intrinsics_as_network_input": False,
                "uses_gate_geometry": True,
                "uses_previous_frames": True,
            },
            "output": {
                "segmentation": True,
                "segmentation_type": "union_auxiliary",
                "instance_segmentation": False,
                "outer_keypoints": True,
                "inner_keypoints": False,
                "pose": True,
                "target_visibility": True,
                "multi_gate": False,
            },
            "complexity": {
                "parameter_count": int(params),
                "trainable_parameter_count": int(trainable),
                "model_size_mb_fp32": float(params * 4 / 1e6),
            },
        },
        "camera": {
            "width_px": int(camera.get("width_px", dcfg.get("native_width", 640))),
            "height_px": int(camera.get("height_px", dcfg.get("native_height", 360))),
            "fx": float(camera.get("fx", dcfg.get("nominal_focal", 320.0))),
            "fy": float(camera.get("fy", dcfg.get("nominal_focal", 320.0))),
            "cx": float(camera.get("cx", dcfg.get("native_width", 640) / 2)),
            "cy": float(camera.get("cy", dcfg.get("native_height", 360) / 2)),
            "hfov_deg": float(camera.get("hfov_deg", 90.0)),
            "vfov_deg": float(camera.get("vfov_deg", 58.715507)),
        },
        "dataset": {
            "dataset_root": dcfg.get("dataset_root"),
            "sources": describe_sources(dcfg),
            "split_manifest_version": dcfg.get("split_manifest_version"),
            "split_strategy": "sequence",
            "evaluation_split": split_name,
            "num_evaluated_frame_steps": int(n_frames),
            "num_pose_supervised_frame_steps": int(n_pose),
        },
        "evaluation": {
            "segmentation": {
                "available": True,
                "type": "union_only",
                "pixel_metrics": {
                    "iou_mean": seg_iou["mean"],
                    "iou_median": seg_iou["median"],
                    "iou_std": seg_iou["std"],
                    "dice_mean": seg_dice["mean"],
                    "dice_median": seg_dice["median"],
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "false_positive_rate": _rate(fp_px, fp_px + tn_px),
                    "false_negative_rate": _rate(fn_px, fn_px + tp_px),
                    "specificity": _rate(tn_px, tn_px + fp_px),
                },
                "instance_metrics": {
                    "available": False,
                    "reason": "vanilla GatePoseNetSingle has a one-channel union auxiliary head"
                },
            },
            "keypoints": {
                "available": True,
                "keypoint_set": "outer_4_only",
                "all_keypoints": {
                    "mean_pixel_error": corners["mean"],
                    "median_pixel_error": corners["median"],
                    "std_pixel_error": corners["std"],
                    "rmse_pixel_error": corners["rmse"],
                    "mean_normalized_error": norm_corners["mean"],
                    "median_normalized_error": norm_corners["median"],
                    "pck": {str(th): _rate(pck_counts[th], pck_total)
                            for th in pck_counts},
                },
                "center": {
                    "mean_pixel_error": _stats(center_errs)["mean"],
                    "median_pixel_error": _stats(center_errs)["median"],
                },
                "inner_keypoints": {
                    "available": False,
                    "reason": "not predicted by the original architecture"
                },
            },
            "pose": {
                "available": True,
                "translation": {
                    "mean_error_m": pos["mean"], "median_error_m": pos["median"],
                    "std_error_m": pos["std"], "rmse_m": pos["rmse"],
                    "p90_error_m": pos["p90"], "p95_error_m": pos["p95"],
                    "p99_error_m": pos["p99"],
                },
                "rotation": {
                    "symmetry_aware": True,
                    "mean_error_deg": rot["mean"], "median_error_deg": rot["median"],
                    "std_error_deg": rot["std"], "p90_error_deg": rot["p90"],
                    "p95_error_deg": rot["p95"], "p99_error_deg": rot["p99"],
                },
                "position_components": {
                    "lateral_error_m": _stats(lateral_errs)["mean"],
                    "vertical_error_m": _stats(vertical_errs)["mean"],
                    "depth_error_m": depth["mean"],
                    "x_error_m": _stats(x_errs)["mean"],
                    "y_error_m": _stats(y_errs)["mean"],
                    "z_error_m": _stats(z_errs)["mean"],
                },
                "threshold_accuracy": {
                    "translation_under_0.05m": _rate(sum(e <= .05 for e in pos_errs), len(pos_errs)),
                    "translation_under_0.10m": _rate(sum(e <= .10 for e in pos_errs), len(pos_errs)),
                    "translation_under_0.25m": _rate(sum(e <= .25 for e in pos_errs), len(pos_errs)),
                    "translation_under_0.50m": _rate(sum(e <= .50 for e in pos_errs), len(pos_errs)),
                    "rotation_under_2deg": _rate(sum(e <= 2 for e in rot_errs), len(rot_errs)),
                    "rotation_under_5deg": _rate(sum(e <= 5 for e in rot_errs), len(rot_errs)),
                    "rotation_under_10deg": _rate(sum(e <= 10 for e in rot_errs), len(rot_errs)),
                },
            },
            "gate_detection": {
                "definition": "single current-target visibility detection, not multi-gate detection",
                "gate_detection_rate": vis_recall,
                "precision": vis_precision,
                "recall": vis_recall,
                "f1": vis_f1,
                "accuracy": _rate(vis_tp + vis_tn, vis_tp + vis_tn + vis_fp + vis_fn),
                "catastrophic_failure_rate": _rate(catastrophic, n_pose),
                "catastrophic_thresholds": {
                    "translation_error_m_gt": catastrophic_pos,
                    "rotation_error_deg_gt": catastrophic_rot,
                    "visible_target_missed": True,
                },
            },
            "temporal": {
                "available": True,
                "tracking_identity_metrics_available": False,
                "reason": "original single-target model has recurrent state but no track-ID head",
            },
            "runtime": {
                "device": str(device),
                "precision": "amp/fp32 per runtime configuration",
                "latency_ms": {
                    "definition": "batch-1 model.step deployment latency",
                    "mean": latency["mean"], "median": latency["median"],
                    "p90": latency["p90"], "p95": latency["p95"],
                    "p99": latency["p99"],
                },
                "evaluation_window_forward_ms": {
                    "definition": "full T-frame model.forward timing per evaluated batch item",
                    "mean": window_latency["mean"], "median": window_latency["median"],
                    "p95": window_latency["p95"],
                },
                "fps": {
                    "mean": (1000.0 / latency["mean"] if latency["mean"] else None),
                    "median": (1000.0 / latency["median"] if latency["median"] else None),
                },
                "gpu": {"peak_vram_mb": peak_vram},
            },
            "robustness": {
                "by_gate_distance_translation_error_m": {
                    k: {"mean": _stats(v)["mean"], "median": _stats(v)["median"], "n": len(v)}
                    for k, v in distance_buckets.items()
                },
                "by_source_domain": by_source,
            },
            "closed_loop": {"available": False},
        },
        "leaderboard": {
            "primary_metric": "translation_median_m",
            "primary_metric_value": pos["median"],
            "secondary_metrics": {
                "rotation_median_deg": rot["median"],
                "union_mask_iou": seg_iou["mean"],
                "target_visibility_recall": vis_recall,
                "catastrophic_failure_rate": _rate(catastrophic, n_pose),
                "inference_latency_ms": latency["mean"],
            },
            "composite_score": None,
            "passes_constraints": None,
        },
        "artifacts": {
            "best_checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "evaluation_json": str(output_dir / "evaluation.json"),
            "per_sample_metrics_csv": str(output_dir / "per_sample_metrics.csv"),
        },
        "reproducibility": {
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "operating_system": platform.platform(),
            "seed": cfg.get("train", {}).get("seed"),
        },
        "notes": {
            "known_limitations": [
                "single target only",
                "outer four corners only",
                "union auxiliary segmentation only",
                "no persistent multi-gate track IDs",
            ],
            "compatibility": "Rich canonical GT is adapted down to the unchanged legacy model interface.",
        },
    }

    report = _safe_json(report)
    with (output_dir / "evaluation.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=False, allow_nan=False)
        f.write("\n")

    fieldnames = [
        "sample_index", "source", "sequence_id", "frame_index", "window_t",
        "gt_visible", "pred_visible",
        "union_iou", "union_dice", "gate_distance_m",
        "translation_error_m", "rotation_error_deg", "depth_error_m",
        "outer_corner_error_px", "catastrophic_failure",
    ]
    with (output_dir / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(per_sample)

    return report
