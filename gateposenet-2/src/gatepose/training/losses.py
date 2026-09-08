from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from gatepose.training.matcher import match_frame_persistent


def _resize_mask(target, hw):
    if tuple(target.shape[-2:]) == tuple(hw): return target
    return F.interpolate(target[:,None].float(),size=hw,mode="nearest")[:,0]


def dice_loss(logits,target):
    target=_resize_mask(target,logits.shape[-2:])
    p=logits.sigmoid().flatten(1); t=target.flatten(1)
    return (1-(2*(p*t).sum(1)+1)/(p.sum(1)+t.sum(1)+1)).mean()


def geodesic(R1,R2):
    R=R1.transpose(-1,-2)@R2; tr=R.diagonal(dim1=-2,dim2=-1).sum(-1)
    return torch.acos(((tr-1)/2).clamp(-0.999999,0.999999))


def c4_rotation_loss(pred_R,gt_R):
    device=pred_R.device; mats=[]
    for a in [0.,math.pi/2,math.pi,3*math.pi/2]:
        c,s=math.cos(a),math.sin(a); mats.append(torch.tensor([[c,-s,0],[s,c,0],[0,0,1]],device=device,dtype=pred_R.dtype))
    return torch.stack([geodesic(pred_R,gt_R@S) for S in mats],0).min(0).values.mean()


def gate_points(device,dtype):
    # gate-local x right, y down, approach face z=-depth/2
    z=-0.13
    return torch.tensor([[-1.35,-1.35,z],[1.35,-1.35,z],[1.35,1.35,z],[-1.35,1.35,z],[-.75,-.75,z],[.75,-.75,z],[.75,.75,z],[-.75,.75,z]],device=device,dtype=dtype)


def project_pose_keypoints(R,t,kin,model_w,model_h):
    pts=gate_points(R.device,R.dtype); pc=(R@pts.T).T+t[None]; z=pc[:,2].clamp_min(1e-4)
    fx=kin[0]*model_w; fy=kin[1]*model_h; cx=kin[2]*(model_w-1); cy=kin[3]*(model_h-1)
    u=fx*pc[:,0]/z+cx; v=fy*pc[:,1]/z+cy
    return torch.stack([u/(model_w-1),v/(model_h-1)],-1)


def compute_losses(outputs:dict,batch:dict,cfg:dict):
    w=cfg["loss"]; dev=outputs["instances"]["object_logits"].device; total=torch.tensor(0.,device=dev); parts={}; B,T,Q=outputs["instances"]["object_logits"].shape
    names=["presence","mask_bce","mask_dice","box","keypoint","kp_conf","kp_state","position","rotation","depth","visible","target","reproject","track"]
    accum={k:[] for k in names}
    for b in range(B):
        track_to_query={}; last_track_emb={}
        for t in range(T):
            gt=batch["frames"][b][t]["instances"]
            pred={"mask_logits":outputs["instances"]["mask_logits"][b,t],"boxes":outputs["instances"]["boxes"][b,t],"keypoints":outputs["instances"]["keypoints"][b,t],"translation":outputs["instances"]["pose"]["translation"][b,t]}
            qi,gi=match_frame_persistent(pred,gt,track_to_query)
            obj_target=torch.zeros(Q,device=dev); target_q=torch.zeros(Q,device=dev)
            if len(qi): obj_target[torch.as_tensor(qi,device=dev)]=1
            accum["presence"].append(F.binary_cross_entropy_with_logits(outputs["instances"]["object_logits"][b,t],obj_target))
            for q,gix in zip(qi,gi):
                g=gt[int(gix)]; target_q[q]=1.0 if g.get("is_target",False) else 0.0
                gm=g["mask"].to(dev)[None]; pm=outputs["instances"]["mask_logits"][b,t,q][None]; gm_small=_resize_mask(gm,pm.shape[-2:]); scale=float(w.get("blind_scale",0.3)) if gm.sum()==0 else 1.0
                accum["mask_bce"].append(F.binary_cross_entropy_with_logits(pm,gm_small)*scale); accum["mask_dice"].append(dice_loss(pm,gm)*scale)
                accum["box"].append(F.l1_loss(outputs["instances"]["boxes"][b,t,q],g["box"].to(dev)))
                valid=g["keypoint_valid"].to(dev); kp_pred=outputs["instances"]["keypoints"][b,t,q]; kp_gt=g["keypoints"].to(dev)
                if valid.any(): accum["keypoint"].append(F.smooth_l1_loss(kp_pred[valid],kp_gt[valid]))
                accum["kp_conf"].append(F.binary_cross_entropy_with_logits(outputs["instances"]["keypoint_logits"][b,t,q],valid.float()))
                accum["kp_state"].append(F.cross_entropy(outputs["instances"]["visibility_logits"][b,t,q],g["visibility"].to(dev)))
                trans=outputs["instances"]["pose"]["translation"][b,t,q]; accum["position"].append(F.smooth_l1_loss(trans,g["translation"].to(dev))*scale)
                Rp=outputs["auxiliary"]["rotation_matrix"][b,t,q]; Rg=g["rotation_matrix"].to(dev); accum["rotation"].append(c4_rotation_loss(Rp[None],Rg[None])*scale)
                accum["depth"].append(F.smooth_l1_loss(outputs["auxiliary"]["depth_log"][b,t,q],torch.log(g["depth"].to(dev).clamp_min(1e-3))))
                accum["visible"].append(F.l1_loss(outputs["auxiliary"]["visible_fraction"][b,t,q],g["visible_fraction"].to(dev)))
                projected=project_pose_keypoints(Rp,trans,batch["intrinsics"][b,t],cfg["camera"]["model_width"],cfg["camera"]["model_height"])
                if valid.any(): accum["reproject"].append(F.smooth_l1_loss(projected[valid],kp_pred[valid]))
                tid=g.get("track_id"); emb=F.normalize(outputs["instances"]["track_embeddings"][b,t,q],dim=-1)
                if tid is not None and tid in last_track_emb: accum["track"].append(1-(emb*last_track_emb[tid]).sum())
                if tid is not None: last_track_emb[tid]=emb.detach()
            accum["target"].append(F.binary_cross_entropy_with_logits(outputs["auxiliary"]["target_logits"][b,t],target_q))
    def mean(name): return torch.stack(accum[name]).mean() if accum[name] else torch.tensor(0.,device=dev)
    mapping={"presence":"w_presence","target":"w_target","mask_bce":"w_mask_bce","mask_dice":"w_mask_dice","box":"w_box","keypoint":"w_keypoint","kp_conf":"w_keypoint_confidence","kp_state":"w_keypoint_state","position":"w_position","rotation":"w_rotation","depth":"w_depth","visible":"w_visible_fraction","reproject":"w_reprojection","track":"w_track"}
    for name,key in mapping.items():
        val=mean(name); parts[name]=val; total=total+float(w.get(key,0.0))*val
    parts["total"]=total
    return total,parts
