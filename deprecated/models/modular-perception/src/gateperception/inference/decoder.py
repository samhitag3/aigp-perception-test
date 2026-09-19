from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class Detection:
    mask_id: int | None
    score: float
    mask: np.ndarray
    box_xyxy_px: list[float]
    track_embedding: np.ndarray
    track_id: str | None = None
    source: str = "model"


def _bbox(mask: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


@torch.no_grad()
def decode_segmentation_frame(
    raw: dict[str, torch.Tensor],
    out_h: int,
    out_w: int,
    object_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    min_area_px: int = 25,
) -> tuple[np.ndarray, list[Detection]]:
    # raw is one batch element: logits [Q], masks [Q,h,w], embeddings [Q,D]
    scores = raw["object_logits"].sigmoid()
    keep = torch.nonzero(scores >= object_threshold, as_tuple=False).flatten()
    if len(keep) == 0:
        return np.zeros((out_h, out_w), dtype=np.uint16), []
    probs = raw["mask_logits"][keep].sigmoid()
    probs = F.interpolate(probs[:, None], size=(out_h, out_w), mode="bilinear", align_corners=False)[:, 0]
    kept_scores = scores[keep]
    weighted = probs * kept_scores[:, None, None]
    winning_score, winner = weighted.max(dim=0)
    winner_prob = torch.gather(probs, 0, winner[None])[0]
    valid = winner_prob >= mask_threshold
    inst = torch.zeros((out_h, out_w), dtype=torch.int32, device=probs.device)
    detections: list[Detection] = []
    next_id = 1
    for local_i, query_i in enumerate(keep.tolist()):
        m = valid & (winner == local_i)
        area = int(m.sum())
        if area < min_area_px:
            continue
        inst[m] = next_id
        mn = m.detach().cpu().numpy().astype(bool)
        detections.append(Detection(
            mask_id=next_id,
            score=float(scores[query_i]),
            mask=mn,
            box_xyxy_px=_bbox(mn),
            track_embedding=raw["track_embeddings"][query_i].detach().cpu().numpy().astype(np.float32),
        ))
        next_id += 1
    return inst.detach().cpu().numpy().astype(np.uint16), detections
