from __future__ import annotations
import math
import torch
from gateperceiver.geometry import rotation_6d_to_matrix, rotation_geodesic_deg


def matched_batch_stats(outputs, targets, matches, threshold=0.5):
    s={"tp":0,"fp":0,"fn":0,"iou_sum":0.,"dice_sum":0.,"pixel_precision_sum":0.,"pixel_recall_sum":0.,"kp_px_sum":0.,"kp_count":0,"translation_sum":0.,"rotation_deg_sum":0.,"matched":0,"frames":0}
    B,T,Q=outputs["pred_logits"].shape
    for b in range(B):
      for t in range(T):
        s["frames"]+=1; scores=outputs["pred_logits"][b,t].sigmoid(); keep=scores>=threshold; N=targets[b][t]["boxes"].shape[0]
        pi,ti=matches[b][t]; matched_kept=[(p,tt) for p,tt in zip(pi.tolist(),ti.tolist()) if bool(keep[p])]
        tp=len(matched_kept); s["tp"]+=tp; s["fp"]+=max(0,int(keep.sum())-tp); s["fn"]+=max(0,N-tp)
        H,W=targets[b][t]["size"].tolist()
        for pidx,tidx in matched_kept:
            pm=outputs["pred_masks"][b,t,pidx].sigmoid()>=0.5; tm=targets[b][t]["masks"][tidx].to(pm.device)>0.5
            inter=(pm&tm).sum().item(); union=(pm|tm).sum().item(); ps=pm.sum().item(); ts=tm.sum().item()
            s["iou_sum"]+=inter/max(1,union); s["dice_sum"]+=(2*inter)/max(1,ps+ts); s["pixel_precision_sum"]+=inter/max(1,ps); s["pixel_recall_sum"]+=inter/max(1,ts)
            pk=outputs["pred_keypoints"][b,t,pidx]; tk=targets[b][t]["keypoints"][tidx].to(pk.device); valid=torch.isfinite(tk).all(-1)
            if valid.any():
                scale=pk.new_tensor([W-1,H-1]); err=torch.linalg.vector_norm((pk-torch.nan_to_num(tk))*scale,dim=-1); s["kp_px_sum"]+=err[valid].sum().item(); s["kp_count"]+=int(valid.sum())
            ptr=outputs["pred_translation"][b,t,pidx]; ttr=targets[b][t]["translation"][tidx].to(ptr.device); s["translation_sum"]+=torch.linalg.vector_norm(ptr-ttr).item()
            pr=rotation_6d_to_matrix(outputs["pred_rotation6d"][b,t,pidx]); tr=rotation_6d_to_matrix(targets[b][t]["rotation6d"][tidx].to(pr.device)); s["rotation_deg_sum"]+=rotation_geodesic_deg(pr[None],tr[None]).item(); s["matched"]+=1
    return s


def reduce_stats(stats_list):
    s={k:sum(x[k] for x in stats_list) for k in stats_list[0]} if stats_list else {}
    tp,fp,fn=s.get("tp",0),s.get("fp",0),s.get("fn",0); m=max(1,s.get("matched",0))
    precision=tp/max(1,tp+fp); recall=tp/max(1,tp+fn)
    return {
      "gate_precision":precision,"gate_recall":recall,"gate_f1":2*precision*recall/max(1e-9,precision+recall),
      "missed_gate_rate":fn/max(1,tp+fn),"false_positive_gates_per_frame":fp/max(1,s.get("frames",1)),
      "mean_instance_iou":s.get("iou_sum",0)/m,"mean_dice":s.get("dice_sum",0)/m,
      "pixel_precision":s.get("pixel_precision_sum",0)/m,"pixel_recall":s.get("pixel_recall_sum",0)/m,
      "mean_keypoint_error_px":s.get("kp_px_sum",0)/max(1,s.get("kp_count",0)),
      "mean_translation_error_m":s.get("translation_sum",0)/m,"mean_rotation_error_deg":s.get("rotation_deg_sum",0)/m,
      "matched_instances":s.get("matched",0),"frames":s.get("frames",0)
    }
