from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from gateperception.data.canonical import CanonicalIndex, KEYPOINT_ORDER


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]

def iou(a,b):
    inter=np.logical_and(a,b).sum(); union=np.logical_or(a,b).sum(); return float(inter/union) if union else 0.0

def rot_error_deg(Tp,Tg):
    Rp=np.asarray(Tp)[:3,:3]; Rg=np.asarray(Tg)[:3,:3]; R=Rp@Rg.T
    c=np.clip((np.trace(R)-1)/2,-1,1); return float(np.degrees(np.arccos(c)))

def pct(values,p): return float(np.percentile(values,p)) if values else None

def safe_mean(v): return float(np.mean(v)) if v else None

def safe_median(v): return float(np.median(v)) if v else None


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",required=True); ap.add_argument("--split",default="test"); ap.add_argument("--predictions",required=True); ap.add_argument("--output",required=True); ap.add_argument("--match-iou",type=float,default=0.5); args=ap.parse_args()
    idx=CanonicalIndex(args.dataset,[args.split]); pred_root=Path(args.predictions); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    pixel_tp=pixel_fp=pixel_fn=pixel_tn=0
    gt_instances=pred_instances=matched_instances=0; count_correct=0; frames_total=0
    inst_ious=[]; kp_errors=[]; kp_norm=[]; visible_err=[]; occluded_err=[]; pose_t=[]; pose_r=[]; reproj=[]; runtimes=[]; catastrophic=0
    pck_counts=defaultdict(int); pck_total=0; rows=[]
    for sid in idx.sequence_ids:
        preds={int(r["frame_index"]):r for r in read_jsonl(pred_root/sid/"frames.jsonl")}
        for gt in idx.sequences[sid]:
            fi=int(gt["frame_index"]); pr=preds.get(fi,{"instances":[],"num_gate_instances":0})
            gtmask_src=np.asarray(Image.open(idx.frame_path(sid,gt["files"]["instance_mask"])))
            pmask_path=pred_root/sid/pr.get("instance_mask","")
            pmask=np.asarray(Image.open(pmask_path)) if pmask_path.exists() else np.zeros_like(gtmask_src)
            src_h,src_w=gtmask_src.shape[:2]; pred_h,pred_w=pmask.shape[:2]
            sx,sy=pred_w/max(1,src_w),pred_h/max(1,src_h)
            if (src_h,src_w)!=(pred_h,pred_w):
                gtmask=cv2.resize(gtmask_src.astype(np.int32),(pred_w,pred_h),interpolation=cv2.INTER_NEAREST).astype(gtmask_src.dtype)
            else:
                gtmask=gtmask_src
            gu=gtmask>0; pu=pmask>0
            pixel_tp+=np.logical_and(gu,pu).sum(); pixel_fp+=np.logical_and(~gu,pu).sum(); pixel_fn+=np.logical_and(gu,~pu).sum(); pixel_tn+=np.logical_and(~gu,~pu).sum()
            gt_gates=[g for g in gt.get("gates",[]) if g.get("mask_id") is not None and np.any(gtmask==int(g["mask_id"]))]
            pred_gates=[p for p in pr.get("instances",[]) if p.get("mask_id") is not None]
            gt_instances+=len(gt_gates); pred_instances+=len(pred_gates); frames_total+=1; count_correct+=int(len(gt_gates)==len(pred_gates))
            runtimes.append(float(pr.get("runtime",{}).get("total_ms",0.0)))
            if gt_gates and pred_gates:
                C=np.ones((len(gt_gates),len(pred_gates)),dtype=float)
                for i,g in enumerate(gt_gates):
                    gm=gtmask==int(g["mask_id"])
                    for j,p in enumerate(pred_gates): C[i,j]=1-iou(gm,pmask==int(p["mask_id"]))
                rr,cc=linear_sum_assignment(C)
            else: rr,cc=np.array([],dtype=int),np.array([],dtype=int)
            matched_gt=set()
            for i,j in zip(rr,cc):
                miou=1-float(C[i,j])
                if miou<args.match_iou: continue
                matched_instances+=1; matched_gt.add(i); inst_ious.append(miou)
                g,p=gt_gates[i],pred_gates[j]
                box=g.get("bounding_boxes",{}).get("amodal_xyxy_px") or g.get("bounding_boxes",{}).get("visible_xyxy_px")
                if box:
                    box_eval=[float(box[0])*sx,float(box[1])*sy,float(box[2])*sx,float(box[3])*sy]
                    diag=math.hypot(box_eval[2]-box_eval[0],box_eval[3]-box_eval[1])
                else:
                    diag=math.hypot(gtmask.shape[1],gtmask.shape[0])
                per_k=[]
                if p.get("keypoints"):
                    for name in KEYPOINT_ORDER:
                        gr=g.get("keypoints_2d",{}).get(name,{}); pp=p["keypoints"].get(name,{})
                        xygt=gr.get("projected_px"); xyp=pp.get("xy_px")
                        if gr.get("projection_valid",False) and xygt is not None and xyp is not None:
                            xygt_eval=np.asarray([float(xygt[0])*sx,float(xygt[1])*sy],dtype=float)
                            e=float(np.linalg.norm(np.asarray(xyp,dtype=float)-xygt_eval)); n=e/max(diag,1e-6)
                            kp_errors.append(e); kp_norm.append(n); per_k.append(e); pck_total+=1
                            for th in (0.01,0.02,0.05,0.10): pck_counts[th]+=int(n<=th)
                            st=gr.get("visibility_state");
                            if st=="visible": visible_err.append(e)
                            elif st=="occluded": occluded_err.append(e)
                te=re=None
                if p.get("pose") and g.get("pose",{}).get("T_camera_gate"):
                    Tp=np.asarray(p["pose"]["T_camera_gate"],dtype=float); Tg=np.asarray(g["pose"]["T_camera_gate"],dtype=float)
                    te=float(np.linalg.norm(Tp[:3,3]-Tg[:3,3])); re=rot_error_deg(Tp,Tg); pose_t.append(te); pose_r.append(re)
                    if p["pose"].get("mean_reprojection_error_px") is not None: reproj.append(float(p["pose"]["mean_reprojection_error_px"]))
                is_cat=(te is not None and te>0.75) or (re is not None and re>25.0)
                catastrophic+=int(is_cat)
                rows.append({"sequence_id":sid,"frame_index":fi,"gt_track_id":g.get("track_id"),"pred_track_id":p.get("track_id"),"iou":miou,"keypoint_mean_error_px":safe_mean(per_k),"translation_error_m":te,"rotation_error_deg":re,"detection_score":p.get("detection_score"),"catastrophic_failure":bool(is_cat),"distance_camera_m":g.get("pose",{}).get("distance_camera_m"),"occlusion_fraction":g.get("visibility",{}).get("occlusion_fraction"),"truncation_fraction":g.get("visibility",{}).get("truncation_fraction"),"view_angle_deg":g.get("pose",{}).get("view_angle_deg")})
            missed=len(gt_gates)-len(matched_gt); catastrophic+=missed
            for i,g in enumerate(gt_gates):
                if i not in matched_gt:
                    rows.append({"sequence_id":sid,"frame_index":fi,"gt_track_id":g.get("track_id"),"pred_track_id":None,"iou":0.0,"keypoint_mean_error_px":None,"translation_error_m":None,"rotation_error_deg":None,"detection_score":None,"catastrophic_failure":True,"distance_camera_m":g.get("pose",{}).get("distance_camera_m"),"occlusion_fraction":g.get("visibility",{}).get("occlusion_fraction"),"truncation_fraction":g.get("visibility",{}).get("truncation_fraction"),"view_angle_deg":g.get("pose",{}).get("view_angle_deg")})
    precision=pixel_tp/max(1,pixel_tp+pixel_fp); recall=pixel_tp/max(1,pixel_tp+pixel_fn); dice=2*pixel_tp/max(1,2*pixel_tp+pixel_fp+pixel_fn); union_iou=pixel_tp/max(1,pixel_tp+pixel_fp+pixel_fn)
    det_precision=matched_instances/max(1,pred_instances); det_recall=matched_instances/max(1,gt_instances); det_f1=2*det_precision*det_recall/max(1e-12,det_precision+det_recall)
    report={
      "schema_version":"1.0.0",
      "evaluation":{"evaluation_split":args.split,"num_sequences":len(idx.sequence_ids),"num_frames":frames_total,"num_gate_instances":gt_instances,
        "segmentation":{"available":True,"pixel_metrics":{"iou_mean":union_iou,"dice_mean":dice,"precision":precision,"recall":recall,"f1":dice,"false_positive_rate":pixel_fp/max(1,pixel_fp+pixel_tn),"false_negative_rate":1-recall},"instance_metrics":{"mean_instance_iou":safe_mean(inst_ious),"gate_instance_precision":det_precision,"gate_instance_recall":det_recall,"gate_instance_f1":det_f1,"instance_count_accuracy":count_correct/max(1,frames_total),"missed_gate_rate":1-det_recall}},
        "keypoints":{"available":bool(kp_errors),"all_keypoints":{"mean_pixel_error":safe_mean(kp_errors),"median_pixel_error":safe_median(kp_errors),"p95_pixel_error":pct(kp_errors,95),"mean_normalized_error":safe_mean(kp_norm),"pck":{"threshold_0.01":pck_counts[0.01]/max(1,pck_total),"threshold_0.02":pck_counts[0.02]/max(1,pck_total),"threshold_0.05":pck_counts[0.05]/max(1,pck_total),"threshold_0.10":pck_counts[0.10]/max(1,pck_total)}},"visible_keypoints":{"mean_pixel_error":safe_mean(visible_err)},"occluded_keypoints":{"mean_pixel_error":safe_mean(occluded_err)}},
        "pose":{"available":bool(pose_t),"translation":{"mean_error_m":safe_mean(pose_t),"median_error_m":safe_median(pose_t),"p90_error_m":pct(pose_t,90),"p95_error_m":pct(pose_t,95),"p99_error_m":pct(pose_t,99)},"rotation":{"mean_error_deg":safe_mean(pose_r),"median_error_deg":safe_median(pose_r),"p90_error_deg":pct(pose_r,90),"p95_error_deg":pct(pose_r,95),"p99_error_deg":pct(pose_r,99)},"geometry":{"mean_reprojection_error_px":safe_mean(reproj)}},
        "gate_detection":{"gate_detection_rate":det_recall,"precision":det_precision,"recall":det_recall,"f1":det_f1,"catastrophic_failure_rate":catastrophic/max(1,gt_instances)},
        "runtime":{"latency_ms":{"mean":safe_mean(runtimes),"median":safe_median(runtimes),"p95":pct(runtimes,95),"p99":pct(runtimes,99)},"fps":{"mean":1000/safe_mean(runtimes) if safe_mean(runtimes) else None}}
      }}
    with open(out/"evaluation.json","w",encoding="utf-8") as f: json.dump(report,f,indent=2)
    fields=list(rows[0].keys()) if rows else ["sequence_id","frame_index"]
    with open(out/"per_sample_metrics.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(json.dumps(report["evaluation"]["gate_detection"],indent=2))

if __name__=="__main__": main()
