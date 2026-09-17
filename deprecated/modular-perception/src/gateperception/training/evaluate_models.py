from __future__ import annotations
from collections import defaultdict
import math
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from gateperception.inference.decoder import decode_segmentation_frame


def _safe_mean(xs): return float(np.mean(xs)) if xs else None

def _safe_median(xs): return float(np.median(xs)) if xs else None


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a,b).sum(); union = np.logical_or(a,b).sum()
    return float(inter/union) if union else 0.0


@torch.no_grad()
def evaluate_segmentation_model(model, loader, device, cfg: dict) -> dict:
    model.eval(); H=int(cfg['dataset'].get('height',360)); W=int(cfg['dataset'].get('width',640))
    obj_th=float(cfg.get('evaluation',{}).get('object_threshold',0.5)); mask_th=float(cfg.get('evaluation',{}).get('mask_threshold',0.5)); match_th=float(cfg.get('evaluation',{}).get('match_iou',0.5))
    tp=fp=fn=tn=0; gt_n=pred_n=matched=frames=count_ok=0; ious=[]
    for batch in loader:
        rgb=batch['rgb'].to(device,non_blocking=True)
        raw=model(rgb,return_all_frames=False)['frames'][-1]
        B=rgb.shape[0]
        for b in range(B):
            r={k:v[b] for k,v in raw.items()}
            pmask,pdets=decode_segmentation_frame(r,H,W,obj_th,mask_th)
            gt=batch['targets'][b][-1]
            gmasks=[m.numpy().astype(bool) for m in gt['masks']]
            g_union=np.any(np.stack(gmasks),axis=0) if gmasks else np.zeros((H,W),bool)
            p_union=pmask>0
            tp+=np.logical_and(g_union,p_union).sum(); fp+=np.logical_and(~g_union,p_union).sum(); fn+=np.logical_and(g_union,~p_union).sum(); tn+=np.logical_and(~g_union,~p_union).sum()
            gt_n+=len(gmasks); pred_n+=len(pdets); frames+=1; count_ok+=int(len(gmasks)==len(pdets))
            if gmasks and pdets:
                C=np.ones((len(gmasks),len(pdets)),dtype=float)
                for i,g in enumerate(gmasks):
                    for j,p in enumerate(pdets): C[i,j]=1-_iou(g,p.mask)
                rr,cc=linear_sum_assignment(C)
                for i,j in zip(rr,cc):
                    val=1-float(C[i,j])
                    if val>=match_th: matched+=1; ious.append(val)
    precision=tp/max(1,tp+fp); recall=tp/max(1,tp+fn); dice=2*tp/max(1,2*tp+fp+fn); pix_iou=tp/max(1,tp+fp+fn)
    ip=matched/max(1,pred_n); ir=matched/max(1,gt_n); i_f1=2*ip*ir/max(1e-12,ip+ir)
    return {
      'segmentation':{
        'available':True,
        'pixel_metrics':{'iou_mean':float(pix_iou),'dice_mean':float(dice),'precision':float(precision),'recall':float(recall),'f1':float(dice),'false_positive_rate':float(fp/max(1,fp+tn)),'false_negative_rate':float(1-recall)},
        'instance_metrics':{'mean_instance_iou':_safe_mean(ious),'median_instance_iou':_safe_median(ious),'gate_instance_precision':float(ip),'gate_instance_recall':float(ir),'gate_instance_f1':float(i_f1),'instance_count_accuracy':float(count_ok/max(1,frames)),'missed_gate_rate':float(1-ir)}
      },
      'gate_detection':{'gate_detection_rate':float(ir),'precision':float(ip),'recall':float(ir),'f1':float(i_f1)}
    }


@torch.no_grad()
def evaluate_keypoint_model(model, loader, device, cfg: dict) -> dict:
    model.eval(); cw,ch=map(float,cfg['model'].get('crop_size',[192,192])); scale=torch.tensor([cw,ch],device=device); diag=math.hypot(cw,ch)
    errors=[]; norms=[]; visible=[]; occluded=[]; vis_correct=vis_total=0; pck=defaultdict(int); total=0
    for batch in loader:
        x=batch['x'].to(device,non_blocking=True); out=model(x); gt=batch['keypoints'].to(device); valid=batch['valid'].to(device); vis=batch['visibility'].to(device)
        err=torch.linalg.vector_norm((out['keypoints']-gt)*scale,dim=-1)
        pred_vis=out['visibility_logits'].argmax(-1); vis_correct+=int((pred_vis==vis).sum()); vis_total+=vis.numel()
        if valid.any():
            vals=err[valid].detach().cpu().tolist(); errors.extend(vals)
            n=(err/diag)[valid].detach().cpu().tolist(); norms.extend(n)
            for xval in n:
                total+=1
                for th in (0.01,0.02,0.05,0.10): pck[th]+=int(xval<=th)
        visible.extend(err[(vis==1)&valid].detach().cpu().tolist()); occluded.extend(err[(vis==2)&valid].detach().cpu().tolist())
    return {
      'keypoints':{
        'available':True,
        'all_keypoints':{'mean_pixel_error':_safe_mean(errors),'median_pixel_error':_safe_median(errors),'mean_normalized_error':_safe_mean(norms),'pck':{'threshold_0.01':pck[0.01]/max(1,total),'threshold_0.02':pck[0.02]/max(1,total),'threshold_0.05':pck[0.05]/max(1,total),'threshold_0.10':pck[0.10]/max(1,total)}},
        'visible_keypoints':{'mean_pixel_error':_safe_mean(visible)},
        'occluded_keypoints':{'mean_pixel_error':_safe_mean(occluded)},
        'visibility_classification':{'accuracy':vis_correct/max(1,vis_total)}
      }
    }
