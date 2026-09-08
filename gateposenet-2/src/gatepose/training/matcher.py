from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def _resize_targets(targets: torch.Tensor, hw: tuple[int,int]) -> torch.Tensor:
    if tuple(targets.shape[-2:]) == tuple(hw):
        return targets
    return F.interpolate(targets[:,None].float(), size=hw, mode="nearest")[:,0]


def dice_cost(pred_logits, targets):
    targets = _resize_targets(targets, pred_logits.shape[-2:])
    p=pred_logits.sigmoid().flatten(1); t=targets.flatten(1)
    inter=torch.einsum('qc,nc->qn',p,t)
    den=p.sum(1)[:,None]+t.sum(1)[None,:]
    return 1-(2*inter+1)/(den+1)


@torch.no_grad()
def match_frame(pred:dict, gt:list[dict], weights=None, query_indices=None, gt_indices=None):
    if not gt: return np.array([],dtype=int),np.array([],dtype=int)
    w={"mask":2.0,"kp":2.0,"pos":2.0,"box":1.0}; w.update(weights or {})
    qidx=np.arange(pred["mask_logits"].shape[0]) if query_indices is None else np.asarray(query_indices,dtype=int)
    gidx=np.arange(len(gt)) if gt_indices is None else np.asarray(gt_indices,dtype=int)
    if len(qidx)==0 or len(gidx)==0: return np.array([],dtype=int),np.array([],dtype=int)
    masks=torch.stack([gt[i]["mask"].to(pred["mask_logits"].device) for i in gidx])
    boxes=torch.stack([gt[i]["box"].to(pred["boxes"].device) for i in gidx])
    kps=torch.stack([gt[i]["keypoints"].to(pred["keypoints"].device) for i in gidx])
    pos=torch.stack([gt[i]["translation"].to(pred["translation"].device) for i in gidx])
    c=w["mask"]*dice_cost(pred["mask_logits"][qidx],masks)
    c+=w["box"]*torch.cdist(pred["boxes"][qidx],boxes,p=1)
    c+=w["kp"]*torch.cdist(pred["keypoints"][qidx].flatten(1),kps.flatten(1),p=1)/16.0
    c+=w["pos"]*torch.cdist(pred["translation"][qidx],pos,p=1)/3.0
    r,cidx=linear_sum_assignment(c.detach().cpu().numpy())
    return qidx[r].astype(int),gidx[cidx].astype(int)


@torch.no_grad()
def match_frame_persistent(pred:dict, gt:list[dict], track_to_query:dict[str,int]):
    """Keep a GT track on its previously assigned query inside a temporal window; Hungarian-match new tracks."""
    forced_q=[]; forced_g=[]; used_q=set(); used_g=set()
    for gi,g in enumerate(gt):
        tid=g.get("track_id")
        q=track_to_query.get(tid) if tid is not None else None
        if q is not None and q < pred["mask_logits"].shape[0] and q not in used_q:
            forced_q.append(q); forced_g.append(gi); used_q.add(q); used_g.add(gi)
    remain_q=[q for q in range(pred["mask_logits"].shape[0]) if q not in used_q]
    remain_g=[g for g in range(len(gt)) if g not in used_g]
    q2,g2=match_frame(pred,gt,query_indices=remain_q,gt_indices=remain_g)
    qs=np.asarray(forced_q+q2.tolist(),dtype=int); gs=np.asarray(forced_g+g2.tolist(),dtype=int)
    for q,gi in zip(qs,gs):
        tid=gt[int(gi)].get("track_id")
        if tid is not None: track_to_query[tid]=int(q)
    return qs,gs
