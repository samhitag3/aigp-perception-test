from __future__ import annotations
import torch
import torch.nn.functional as F
from gateperceiver.geometry import generalized_box_iou_cxcywh, rotation_6d_to_matrix
from .matcher import match_batch


def dice_loss(logits, targets):
    p=logits.sigmoid().flatten(1); t=targets.flatten(1)
    return (1-(2*(p*t).sum(1)+1)/(p.sum(1)+t.sum(1)+1)).mean()


def compute_losses(outputs: dict, targets: list[list[dict]], cfg: dict):
    w=cfg["loss"]; matches=match_batch(outputs,targets,w)
    B,T,Q=outputs["pred_logits"].shape
    obj_target=torch.zeros_like(outputs["pred_logits"])
    loss_mask=outputs["pred_logits"].new_tensor(0.); loss_box=loss_mask.clone(); loss_giou=loss_mask.clone(); loss_kp=loss_mask.clone(); loss_vis=loss_mask.clone(); loss_tr=loss_mask.clone(); loss_rot=loss_mask.clone(); n=0
    matched_emb=[]; matched_tracks=[]
    for b in range(B):
      for t in range(T):
        pi,ti=matches[b][t]
        if len(pi)==0: continue
        pi=pi.to(outputs["pred_logits"].device); ti=ti.to(outputs["pred_logits"].device); tar=targets[b][t]
        obj_target[b,t,pi]=1; n+=len(pi)
        pm=outputs["pred_masks"][b,t,pi]; tm=tar["masks"][ti].to(pm.device)
        loss_mask += F.binary_cross_entropy_with_logits(pm,tm)+dice_loss(pm,tm)
        pb=outputs["pred_boxes"][b,t,pi]; tb=tar["boxes"][ti].to(pb.device)
        loss_box += F.l1_loss(pb,tb); loss_giou += (1-generalized_box_iou_cxcywh(pb,tb)).mean()
        pk=outputs["pred_keypoints"][b,t,pi]; tk=tar["keypoints"][ti].to(pk.device); valid=torch.isfinite(tk).all(-1)
        if valid.any(): loss_kp += (pk-torch.nan_to_num(tk)).abs().sum(-1)[valid].mean()
        pv=outputs["pred_visibility"][b,t,pi].reshape(-1,4); tv=tar["visibility"][ti].to(pv.device).reshape(-1); loss_vis += F.cross_entropy(pv,tv)
        ptr=outputs["pred_translation"][b,t,pi]; ttr=tar["translation"][ti].to(ptr.device); loss_tr += F.smooth_l1_loss(ptr,ttr)
        pr=rotation_6d_to_matrix(outputs["pred_rotation6d"][b,t,pi]); tr=rotation_6d_to_matrix(tar["rotation6d"][ti].to(pr.device)); loss_rot += F.mse_loss(pr,tr)
        for pidx,tidx in zip(pi.tolist(),ti.tolist()): matched_emb.append(outputs["track_embeddings"][b,t,pidx]); matched_tracks.append((b,tar["track_ids"][tidx]))
    denom=max(1,B*T)
    loss_obj=F.binary_cross_entropy_with_logits(outputs["pred_logits"],obj_target,pos_weight=torch.tensor(float(w.get("object_pos_weight",2.0)),device=obj_target.device))
    parts={"object":loss_obj,"mask":loss_mask/denom,"box":loss_box/denom,"giou":loss_giou/denom,"keypoint":loss_kp/denom,"visibility":loss_vis/denom,"translation":loss_tr/denom,"rotation":loss_rot/denom}
    # simple supervised contrastive tracking loss across matched query embeddings within batch/windows
    track=loss_obj.new_tensor(0.)
    if len(matched_emb)>1:
        E=torch.stack(matched_emb); sim=E@E.T; terms=[]
        for i in range(len(matched_tracks)):
            for j in range(i+1,len(matched_tracks)):
                same=matched_tracks[i]==matched_tracks[j]
                terms.append((1-sim[i,j]) if same else F.relu(sim[i,j]-float(w.get("track_negative_margin",0.2))))
        if terms: track=torch.stack(terms).mean()
    parts["track"]=track
    total=sum(float(w.get(f"{k}_weight",0.0))*v for k,v in parts.items())
    parts["total"]=total
    return parts,matches
