from __future__ import annotations
import torch
import torch.nn.functional as F


def keypoint_loss(outputs: dict[str, torch.Tensor], batch: dict, cfg: dict) -> tuple[torch.Tensor, dict[str, float]]:
    pred = outputs["keypoints"]
    gt = batch["keypoints"].to(pred.device)
    valid = batch["valid"].to(pred.device)
    vis = batch["visibility"].to(pred.device)
    if valid.any():
        coord = F.smooth_l1_loss(pred[valid], gt[valid], beta=float(cfg.get("smooth_l1_beta", 0.05)))
    else:
        coord = pred.new_tensor(0.0)
    visibility = F.cross_entropy(outputs["visibility_logits"].reshape(-1, 4), vis.reshape(-1))
    total = float(cfg.get("coord_weight", 5.0)) * coord + float(cfg.get("visibility_weight", 1.0)) * visibility
    return total, {"coord": float(coord.detach()), "visibility": float(visibility.detach()), "total": float(total.detach())}
