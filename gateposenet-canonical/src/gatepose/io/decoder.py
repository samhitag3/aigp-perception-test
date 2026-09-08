from __future__ import annotations
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from gatepose.models.geometry import rotation_6d_to_matrix
from gatepose.utils.camera import inverse_points
KP_NAMES=["outer_tl","outer_tr","outer_br","outer_bl","inner_tl","inner_tr","inner_br","inner_bl"]
VIS_NAMES=["visible","occluded","out_of_frame","behind_camera","invalid"]

def matrix_to_quaternion_xyzw(R: np.ndarray):
    import math
    t=np.trace(R)
    if t>0:
        s=math.sqrt(t+1.0)*2; qw=0.25*s; qx=(R[2,1]-R[1,2])/s; qy=(R[0,2]-R[2,0])/s; qz=(R[1,0]-R[0,1])/s
    else:
        i=int(np.argmax(np.diag(R)))
        if i==0: s=math.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2; qw=(R[2,1]-R[1,2])/s; qx=.25*s; qy=(R[0,1]+R[1,0])/s; qz=(R[0,2]+R[2,0])/s
        elif i==1: s=math.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2; qw=(R[0,2]-R[2,0])/s; qx=(R[0,1]+R[1,0])/s; qy=.25*s; qz=(R[1,2]+R[2,1])/s
        else: s=math.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2; qw=(R[1,0]-R[0,1])/s; qx=(R[0,2]+R[2,0])/s; qy=(R[1,2]+R[2,1])/s; qz=.25*s
    return [float(qx),float(qy),float(qz),float(qw)]

def decode_frame(frame_outputs:dict, tr, source_size, cfg:dict, query_track_ids=None):
    dec=cfg["decoder"]; model_w=int(cfg["camera"]["model_width"]); model_h=int(cfg["camera"]["model_height"])
    obj=frame_outputs["instances"]["object_logits"].sigmoid(); low_masks=frame_outputs["instances"]["mask_logits"]; masks=F.interpolate(low_masks[:,None],size=(model_h,model_w),mode="bilinear",align_corners=False)[:,0].sigmoid(); keep=torch.nonzero(obj>=float(dec.get("object_confidence_threshold",0.5))).flatten()
    src_w,src_h=source_size
    if len(keep)==0: return np.zeros((src_h,src_w),np.uint16),[]
    scoremaps=obj[keep,None,None]*masks[keep]; _,winners=scoremaps.max(0); probs=masks[keep]; inst_mask=np.zeros((model_h,model_w),np.uint16); instances=[]
    for local,q in enumerate(keep.tolist()):
        m=(winners.cpu().numpy()==local)&(probs[local].cpu().numpy()>=float(dec.get("mask_threshold",0.5)))
        if int(m.sum())<int(dec.get("min_area_px",20)): continue
        mask_id=len(instances)+1; inst_mask[m]=mask_id; ys,xs=np.nonzero(m); box_model=np.array([xs.min(),ys.min(),xs.max(),ys.max()],np.float32); box_src=inverse_points(box_model.reshape(2,2),tr).reshape(-1); box_src[[0,2]]=np.clip(box_src[[0,2]],0,src_w-1); box_src[[1,3]]=np.clip(box_src[[1,3]],0,src_h-1)
        kp_norm=frame_outputs["instances"]["keypoints"][q].detach().cpu().numpy(); kp_model=kp_norm*np.array([model_w-1,model_h-1],np.float32); kp_src=inverse_points(kp_model,tr); kp_conf=frame_outputs["instances"]["keypoint_logits"][q].sigmoid().detach().cpu().numpy(); vis=frame_outputs["instances"]["visibility_logits"][q].argmax(-1).detach().cpu().numpy(); t=frame_outputs["instances"]["pose"]["translation"][q].detach().cpu().numpy(); r6=frame_outputs["instances"]["pose"]["rotation"][q:q+1]; R=rotation_6d_to_matrix(r6)[0].detach().cpu().numpy(); T=np.eye(4,dtype=np.float32); T[:3,:3]=R; T[:3,3]=t
        kpd={name:{"xy_px":[float(kp_src[i,0]),float(kp_src[i,1])],"confidence":float(kp_conf[i]),"visibility":VIS_NAMES[int(vis[i])]} for i,name in enumerate(KP_NAMES)}; track=(query_track_ids[q] if query_track_ids is not None else f"track_{q:04d}")
        instances.append({"mask_id":mask_id,"track_id":track,"detection_score":float(obj[q]),"mask_score":float(masks[q][m].mean().item()),"box_xyxy_px":[float(x) for x in box_src],"visible_area_px":int(m.sum()),"keypoints":kpd,"pose":{"T_camera_gate":T.tolist(),"translation_camera_m":[float(x) for x in t],"quaternion_camera_xyzw":matrix_to_quaternion_xyzw(R),"distance_camera_m":float(np.linalg.norm(t)),"depth_camera_z_m":float(t[2]),"confidence":None}})
    y0,y1=tr.pad_y,tr.pad_y+int(round(tr.src_h*tr.scale)); x0,x1=tr.pad_x,tr.pad_x+int(round(tr.src_w*tr.scale)); cropped=inst_mask[y0:y1,x0:x1]; src=cv2.resize(cropped,(src_w,src_h),interpolation=cv2.INTER_NEAREST).astype(np.uint16)
    # visible_area_px must correspond to persisted source mask
    for inst in instances: inst["visible_area_px"]=int((src==inst["mask_id"]).sum())
    return src,instances
