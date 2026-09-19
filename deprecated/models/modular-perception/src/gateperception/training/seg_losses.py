from __future__ import annotations
from typing import Any
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def dice_cost_from_logits(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # pred [Q,H,W], target [N,H,W]
    p = pred.sigmoid().flatten(1)
    t = target.float().flatten(1)
    inter = torch.einsum("qc,nc->qn", p, t)
    denom = p.sum(1)[:, None] + t.sum(1)[None, :]
    return 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)


def bce_mask_cost(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    q = pred.shape[0]
    n = target.shape[0]
    if n == 0:
        return pred.new_zeros((q, 0))
    p = pred[:, None].expand(q, n, -1, -1)
    t = target[None].float().expand(q, n, -1, -1)
    return F.binary_cross_entropy_with_logits(p, t, reduction="none").mean(dim=(-1, -2))


def match_frame(pred: dict[str, torch.Tensor], target: dict[str, Any], cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    n = target["masks"].shape[0]
    q = pred["object_logits"].shape[0]
    if n == 0:
        z = torch.empty((0,), dtype=torch.long, device=pred["object_logits"].device)
        return z, z
    gt_masks = target["masks"].to(pred["mask_logits"].device)
    if pred["mask_logits"].shape[-2:] != gt_masks.shape[-2:]:
        gt_masks_small = F.interpolate(gt_masks[:, None].float(), size=pred["mask_logits"].shape[-2:], mode="nearest")[:, 0] > 0.5
    else:
        gt_masks_small = gt_masks
    gt_boxes = target["boxes"].to(pred["boxes"].device)
    cost = (
        float(cfg.get("cost_mask_bce", 1.0)) * bce_mask_cost(pred["mask_logits"], gt_masks_small)
        + float(cfg.get("cost_mask_dice", 2.0)) * dice_cost_from_logits(pred["mask_logits"], gt_masks_small)
        + float(cfg.get("cost_box", 1.0)) * torch.cdist(pred["boxes"], gt_boxes, p=1)
        - float(cfg.get("cost_object", 0.5)) * pred["object_logits"].sigmoid()[:, None]
    )
    rr, cc = linear_sum_assignment(cost.detach().cpu().numpy())
    return (
        torch.as_tensor(rr, dtype=torch.long, device=pred["object_logits"].device),
        torch.as_tensor(cc, dtype=torch.long, device=pred["object_logits"].device),
    )


def frame_loss(pred: dict[str, torch.Tensor], targets: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float], list[tuple[torch.Tensor, torch.Tensor]]]:
    b, q = pred["object_logits"].shape
    obj_targets = torch.zeros_like(pred["object_logits"])
    mask_bce = pred["object_logits"].new_tensor(0.0)
    mask_dice = pred["object_logits"].new_tensor(0.0)
    box_l1 = pred["object_logits"].new_tensor(0.0)
    matchings: list[tuple[torch.Tensor, torch.Tensor]] = []
    matched_total = 0
    for i in range(b):
        p = {k: v[i] for k, v in pred.items()}
        pi, gi = match_frame(p, targets[i], cfg)
        matchings.append((pi, gi))
        if len(pi) == 0:
            continue
        obj_targets[i, pi] = 1.0
        gt_masks = targets[i]["masks"].to(p["mask_logits"].device)[gi]
        if gt_masks.shape[-2:] != p["mask_logits"][pi].shape[-2:]:
            gt_masks = F.interpolate(gt_masks[:, None].float(), size=p["mask_logits"][pi].shape[-2:], mode="nearest")[:, 0]
        else:
            gt_masks = gt_masks.float()
        pm = p["mask_logits"][pi]
        mask_bce = mask_bce + F.binary_cross_entropy_with_logits(pm, gt_masks, reduction="mean")
        probs = pm.sigmoid().flatten(1)
        gt_flat = gt_masks.flatten(1)
        dice = 1.0 - (2 * (probs * gt_flat).sum(1) + 1) / (probs.sum(1) + gt_flat.sum(1) + 1)
        mask_dice = mask_dice + dice.mean()
        box_l1 = box_l1 + F.l1_loss(p["boxes"][pi], targets[i]["boxes"].to(pm.device)[gi])
        matched_total += 1
    denom = max(1, matched_total)
    mask_bce = mask_bce / denom
    mask_dice = mask_dice / denom
    box_l1 = box_l1 / denom
    pos_weight = torch.tensor(float(cfg.get("object_pos_weight", 2.0)), device=obj_targets.device)
    obj = F.binary_cross_entropy_with_logits(pred["object_logits"], obj_targets, pos_weight=pos_weight)
    total = (
        float(cfg.get("object_weight", 1.0)) * obj
        + float(cfg.get("mask_bce_weight", 2.0)) * mask_bce
        + float(cfg.get("mask_dice_weight", 3.0)) * mask_dice
        + float(cfg.get("box_weight", 1.0)) * box_l1
    )
    stats = {"object": float(obj.detach()), "mask_bce": float(mask_bce.detach()), "mask_dice": float(mask_dice.detach()), "box_l1": float(box_l1.detach())}
    return total, stats, matchings


def temporal_track_loss(
    frames: list[dict[str, torch.Tensor]],
    targets_bt: list[list[dict[str, Any]]],
    matchings: list[list[tuple[torch.Tensor, torch.Tensor]]],
    margin: float = 0.2,
) -> torch.Tensor:
    if len(frames) < 2:
        return frames[0]["object_logits"].new_tensor(0.0)
    losses = []
    b = frames[0]["object_logits"].shape[0]
    for t in range(1, len(frames)):
        for bi in range(b):
            p0, g0 = matchings[t - 1][bi]
            p1, g1 = matchings[t][bi]
            if len(p0) == 0 or len(p1) == 0:
                continue
            map0 = {targets_bt[bi][t - 1]["track_ids"][int(gi)]: int(pi) for pi, gi in zip(p0, g0)}
            map1 = {targets_bt[bi][t]["track_ids"][int(gi)]: int(pi) for pi, gi in zip(p1, g1)}
            shared = sorted(set(map0) & set(map1))
            for tid in shared:
                e0 = frames[t - 1]["track_embeddings"][bi, map0[tid]]
                e1 = frames[t]["track_embeddings"][bi, map1[tid]]
                losses.append(1.0 - F.cosine_similarity(e0[None], e1[None]).mean())
            # Encourage distinct simultaneous gates to have separated embeddings.
            tids = list(map1)
            for i in range(len(tids)):
                for j in range(i + 1, len(tids)):
                    e1 = frames[t]["track_embeddings"][bi, map1[tids[i]]]
                    e2 = frames[t]["track_embeddings"][bi, map1[tids[j]]]
                    sim = F.cosine_similarity(e1[None], e2[None]).mean()
                    losses.append(F.relu(sim - margin))
    if not losses:
        return frames[0]["object_logits"].new_tensor(0.0)
    return torch.stack(losses).mean()


def segmentation_loss(outputs: dict[str, Any], targets_bt: list[list[dict[str, Any]]], cfg: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    frames = outputs["frames"]
    t_count = len(frames)
    all_losses = []
    matchings_by_t = []
    accum = {"object": 0.0, "mask_bce": 0.0, "mask_dice": 0.0, "box_l1": 0.0}
    for t, pred in enumerate(frames):
        targets_t = [sample[t] for sample in targets_bt]
        loss_t, stats_t, match_t = frame_loss(pred, targets_t, cfg)
        all_losses.append(loss_t)
        matchings_by_t.append(match_t)
        for k in accum:
            accum[k] += stats_t[k] / t_count
    base = torch.stack(all_losses).mean()
    tr = temporal_track_loss(frames, targets_bt, matchings_by_t, margin=float(cfg.get("track_margin", 0.2)))
    total = base + float(cfg.get("track_weight", 0.25)) * tr
    accum["track"] = float(tr.detach())
    accum["total"] = float(total.detach())
    return total, accum
