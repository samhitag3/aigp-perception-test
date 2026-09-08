from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a,b):
    inter=np.logical_and(a,b).sum(); union=np.logical_or(a,b).sum(); return float(inter/union) if union else 1.0


def match_instances(pred_mask,gt_mask):
    pids=[int(x) for x in np.unique(pred_mask) if x!=0]; gids=[int(x) for x in np.unique(gt_mask) if x!=0]
    if not pids or not gids: return [],pids,gids
    C=np.ones((len(pids),len(gids)),np.float32)
    for i,p in enumerate(pids):
        for j,g in enumerate(gids): C[i,j]=1-iou(pred_mask==p,gt_mask==g)
    r,c=linear_sum_assignment(C); matches=[(pids[i],gids[j],1-float(C[i,j])) for i,j in zip(r,c)]
    return matches,pids,gids
