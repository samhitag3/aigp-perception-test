from __future__ import annotations
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def hungarian_match_frame(pred: dict, target: dict, weights: dict) -> tuple[torch.Tensor,torch.Tensor]:
    Q=pred["logits"].shape[0]; N=target["boxes"].shape[0]
    if N==0: return torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long)
    obj_cost=-pred["logits"].sigmoid()[:,None].expand(Q,N)
    box_cost=torch.cdist(pred["boxes"],target["boxes"],p=1)
    # Keypoint cost only where target projection is valid.
    pk=pred["keypoints"][:,None]  # Q,1,8,2
    tk=target["keypoints"][None]  # 1,N,8,2
    valid=torch.isfinite(tk).all(-1)
    diff=torch.nan_to_num((pk-tk).abs().sum(-1),nan=0.0)
    kp_cost=(diff*valid).sum(-1)/valid.sum(-1).clamp_min(1)
    # Low-resolution mask cost for speed.
    pm=F.interpolate(pred["masks"][:,None],size=(64,64),mode="bilinear",align_corners=False)[:,0].sigmoid()
    tm=F.interpolate(target["masks"][:,None],size=(64,64),mode="nearest")[:,0]
    pflat=pm.flatten(1); tflat=tm.flatten(1)
    inter=torch.einsum("qc,nc->qn",pflat,tflat)
    dice=1-(2*inter+1)/(pflat.sum(1)[:,None]+tflat.sum(1)[None,:]+1)
    C=(weights.get("match_object",1)*obj_cost+weights.get("match_box",2)*box_cost+weights.get("match_keypoint",1)*kp_cost+weights.get("match_mask",2)*dice)
    r,c=linear_sum_assignment(C.detach().cpu().numpy())
    return torch.as_tensor(r,dtype=torch.long),torch.as_tensor(c,dtype=torch.long)


def match_batch(outputs: dict, targets: list[list[dict]], weights: dict):
    B,T,Q=outputs["pred_logits"].shape; all_matches=[]
    for b in range(B):
        row=[]
        for t in range(T):
            pred={"logits":outputs["pred_logits"][b,t],"boxes":outputs["pred_boxes"][b,t],"keypoints":outputs["pred_keypoints"][b,t],"masks":outputs["pred_masks"][b,t]}
            row.append(hungarian_match_frame(pred,targets[b][t],weights))
        all_matches.append(row)
    return all_matches
