from __future__ import annotations
import numpy as np
import torch
from gateperceiver.geometry import box_cxcywh_to_xyxy, rotation_6d_to_matrix
from gateperceiver.contracts import KEYPOINT_NAMES, INDEX_TO_VISIBILITY


@torch.no_grad()
def decode_frame(outputs: dict, b: int, t: int, cfg: dict):
    d=cfg.get("decode",{}); det_thr=float(d.get("detection_threshold",0.5)); mask_thr=float(d.get("mask_threshold",0.5))
    scores=outputs["pred_logits"][b,t].sigmoid(); keep=torch.where(scores>=det_thr)[0]
    H,W=outputs["pred_masks"].shape[-2:]
    if len(keep)==0: return np.zeros((H,W),np.uint16),[]
    probs=outputs["pred_masks"][b,t,keep].sigmoid(); weighted=probs*scores[keep,None,None]; winner=weighted.argmax(0); bestp=probs.gather(0,winner[None])[0]
    inst=np.zeros((H,W),np.uint16); instances=[]
    boxes=box_cxcywh_to_xyxy(outputs["pred_boxes"][b,t,keep],W,H)
    for j,q in enumerate(keep.tolist(),start=1):
        pix=(winner==(j-1)) & (bestp>=mask_thr)
        if not pix.any(): continue
        inst[pix.cpu().numpy()]=j
        kp=outputs["pred_keypoints"][b,t,q]; vis=outputs["pred_visibility"][b,t,q].argmax(-1)
        kpo={}
        for k,name in enumerate(KEYPOINT_NAMES):
            kpo[name]={"xy_px":[float(kp[k,0]*(W-1)),float(kp[k,1]*(H-1))],"confidence":None,"visibility":INDEX_TO_VISIBILITY[int(vis[k])]}
        R=rotation_6d_to_matrix(outputs["pred_rotation6d"][b,t,q]); tr=outputs["pred_translation"][b,t,q]
        T=torch.eye(4,device=R.device); T[:3,:3]=R; T[:3,3]=tr
        instances.append({"mask_id":j,"track_id":None,"detection_score":float(scores[q]),"mask_score":float(probs[j-1][pix].mean()),"box_xyxy_px":[float(x) for x in boxes[j-1]],"visible_area_px":int(pix.sum()),"keypoints":kpo,"pose":{"T_camera_gate":T.cpu().tolist(),"translation_camera_m":tr.cpu().tolist(),"distance_camera_m":float(torch.linalg.vector_norm(tr)),"depth_camera_z_m":float(tr[2]),"confidence":None},"track_embedding":outputs["track_embeddings"][b,t,q].cpu().tolist()})
    return inst,instances
