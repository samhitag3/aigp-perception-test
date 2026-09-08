from __future__ import annotations
import torch


def frame_row(outputs, targets, matches, b, t, meta, threshold=0.5):
    scores=outputs["pred_logits"][b,t].sigmoid(); keep=scores>=threshold; tar=targets[b][t]; pi,ti=matches[b][t]
    pairs=[(p,gt) for p,gt in zip(pi.tolist(),ti.tolist()) if bool(keep[p])]
    ious=[]; kp_errors=[]; tr_errors=[]
    H,W=tar["size"].tolist()
    for pidx,tidx in pairs:
        pm=outputs["pred_masks"][b,t,pidx].sigmoid()>=0.5; tm=tar["masks"][tidx].to(pm.device)>0.5; inter=(pm&tm).sum().item(); union=(pm|tm).sum().item(); ious.append(inter/max(1,union))
        pk=outputs["pred_keypoints"][b,t,pidx]; tk=tar["keypoints"][tidx].to(pk.device); valid=torch.isfinite(tk).all(-1)
        if valid.any():
            scale=pk.new_tensor([W-1,H-1]); kp_errors.extend(torch.linalg.vector_norm((pk-torch.nan_to_num(tk))*scale,dim=-1)[valid].detach().cpu().tolist())
        tr_errors.append(float(torch.linalg.vector_norm(outputs["pred_translation"][b,t,pidx]-tar["translation"][tidx].to(outputs["pred_translation"].device))))
    tp=len(pairs); pred=int(keep.sum()); gt=tar["boxes"].shape[0]
    return {"sequence_id":meta[b]["sequence_id"],"frame_index":meta[b]["frame_indices"][t],"timestamp_s":meta[b]["timestamps"][t],"gt_gate_count":gt,"pred_gate_count":pred,"tp":tp,"fp":max(0,pred-tp),"fn":max(0,gt-tp),"mean_iou":sum(ious)/len(ious) if ious else None,"mean_keypoint_error_px":sum(kp_errors)/len(kp_errors) if kp_errors else None,"mean_translation_error_m":sum(tr_errors)/len(tr_errors) if tr_errors else None,"catastrophic_failure":bool(gt>0 and tp==0)}
