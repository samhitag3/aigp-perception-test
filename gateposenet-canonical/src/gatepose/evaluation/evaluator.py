from __future__ import annotations
from pathlib import Path
import csv,json,time,math
import cv2
import numpy as np
import torch
from gatepose.io.decoder import decode_frame,VIS_NAMES
from gatepose.evaluation.metrics import match_instances
from gatepose.utils.camera import inverse_points


def pct(x,p): return float(np.percentile(x,p)) if x else None

def c4_rot_error_deg(Rp,Rg):
    vals=[]
    for a in [0,math.pi/2,math.pi,3*math.pi/2]:
        c,s=math.cos(a),math.sin(a); S=np.array([[c,-s,0],[s,c,0],[0,0,1]],np.float32); R=Rp.T@(Rg@S); v=np.clip((np.trace(R)-1)/2,-1,1); vals.append(math.degrees(math.acos(v)))
    return min(vals)

def evaluate(model,loader,cfg,run_dir,device,save_predictions=False):
    out=Path(run_dir); out.mkdir(parents=True,exist_ok=True); rows=[]; ious=[]; kp_err=[]; pos_err=[]; rot_err=[]; depth_err=[]; lateral_err=[]; vis_ok=vis_n=0; tp=fp=fn=0; id_switches=0; last_pred_track={}; all_times=[]
    pred_dir=out/"predictions"; (pred_dir/"instance_masks").mkdir(parents=True,exist_ok=True)
    frames_fp=(pred_dir/"frames.jsonl").open("w",encoding="utf-8") if save_predictions else None
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images=batch["images"].to(device); ego=batch["ego"].to(device); kin=batch["intrinsics"].to(device)
            if device.type=='cuda': torch.cuda.synchronize()
            st=time.perf_counter(); raw=model(images,ego,kin)
            if device.type=='cuda': torch.cuda.synchronize()
            infer_ms=(time.perf_counter()-st)*1000; B,T=images.shape[:2]; all_times.append(infer_ms/max(B,1))
            t=T-1
            for b in range(B):
                f=batch["frames"][b][t]
                frame_raw={"instances":{k:v[b,t] for k,v in raw["instances"].items() if k!='pose'}}; frame_raw["instances"]["pose"]={k:v[b,t] for k,v in raw["instances"]["pose"].items()}
                mask,inst=decode_frame(frame_raw,f["letterbox"],f["source_size"],cfg)
                gt_model=np.zeros((cfg["camera"]["model_height"],cfg["camera"]["model_width"]),np.uint16)
                for gi,g in enumerate(f["instances"],1): gt_model[g["mask"].numpy()>0.5]=gi
                tr=f["letterbox"]; crop=gt_model[tr.pad_y:tr.pad_y+int(round(tr.src_h*tr.scale)),tr.pad_x:tr.pad_x+int(round(tr.src_w*tr.scale))]; gt=cv2.resize(crop,f["source_size"],interpolation=cv2.INTER_NEAREST)
                matches,pids,gids=match_instances(mask,gt); good=[m for m in matches if m[2]>=0.5]; tp+=len(good); fp+=max(0,len(pids)-len(good)); fn+=max(0,len(gids)-len(good)); ious += [m[2] for m in matches]
                frame_kp=[]; frame_pos=[]
                by_mid={x['mask_id']:x for x in inst}
                for pid,gid,miou in good:
                    pr=by_mid.get(pid); gg=f["instances"][gid-1]
                    if pr is None: continue
                    # keypoint error in original source pixels
                    kpgt=gg['keypoints'].numpy()*np.array([cfg['camera']['model_width']-1,cfg['camera']['model_height']-1],np.float32); kpgt=inverse_points(kpgt,tr); valid=gg['keypoint_valid'].numpy()
                    for j,name in enumerate(pr['keypoints']):
                        if valid[j]:
                            e=float(np.linalg.norm(np.asarray(pr['keypoints'][name]['xy_px'])-kpgt[j])); kp_err.append(e); frame_kp.append(e)
                        vis_pred=VIS_NAMES.index(pr['keypoints'][name]['visibility']); vis_gt=int(gg['visibility'][j]); vis_ok += int(vis_pred==vis_gt); vis_n += 1
                    pt=np.asarray(pr['pose']['translation_camera_m'],np.float32); pg=gg['translation'].numpy(); pe=float(np.linalg.norm(pt-pg)); pos_err.append(pe); frame_pos.append(pe); depth_err.append(float(abs(pt[2]-pg[2]))); lateral_err.append(float(np.linalg.norm(pt[:2]-pg[:2])))
                    Rp=np.asarray(pr['pose']['T_camera_gate'],np.float32)[:3,:3]; Rg=gg['rotation_matrix'].numpy(); rot_err.append(c4_rot_error_deg(Rp,Rg))
                    tid=gg.get('track_id'); pred_tid=pr.get('track_id')
                    if tid is not None and pred_tid is not None:
                        key=(batch['sequence_id'][b],tid)
                        if key in last_pred_track and last_pred_track[key]!=pred_tid: id_switches+=1
                        last_pred_track[key]=pred_tid
                rows.append({"sequence_id":batch["sequence_id"][b],"frame_index":f["frame_index"],"instance_iou_mean":float(np.mean([m[2] for m in matches])) if matches else 0.0,"pred_count":len(inst),"gt_count":len(gids),"keypoint_error_px_mean":float(np.mean(frame_kp)) if frame_kp else None,"translation_error_m_mean":float(np.mean(frame_pos)) if frame_pos else None,"inference_ms":infer_ms/max(B,1)})
                if save_predictions:
                    rel=f"instance_masks/{batch['sequence_id'][b]}_{f['frame_index']:06d}.png"; cv2.imwrite(str(pred_dir/rel),mask)
                    rec={"frame_index":f['frame_index'],"timestamp_s":f['timestamp_ns']*1e-9,"instance_mask":rel,"num_gate_instances":len(inst),"instances":inst,"runtime":{"inference_ms":infer_ms/max(B,1),"postprocess_ms":None,"total_ms":None}}
                    frames_fp.write(json.dumps(rec,separators=(',',':'))+'\n')
    if frames_fp: frames_fp.close()
    prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1); f1=2*prec*rec/max(prec+rec,1e-12)
    result={
      "schema_version":"1.0.0",
      "evaluation":{
        "segmentation":{"mean_instance_iou":float(np.mean(ious)) if ious else 0.0,"median_instance_iou":float(np.median(ious)) if ious else 0.0,"gate_precision":prec,"gate_recall":rec,"gate_f1":f1},
        "keypoints":{"mean_pixel_error":float(np.mean(kp_err)) if kp_err else None,"median_pixel_error":float(np.median(kp_err)) if kp_err else None,"p95_pixel_error":pct(kp_err,95),"visibility_accuracy":vis_ok/max(vis_n,1)},
        "pose":{"translation":{"mean_error_m":float(np.mean(pos_err)) if pos_err else None,"median_error_m":float(np.median(pos_err)) if pos_err else None,"p95_error_m":pct(pos_err,95)},"rotation":{"mean_error_deg":float(np.mean(rot_err)) if rot_err else None,"median_error_deg":float(np.median(rot_err)) if rot_err else None,"p95_error_deg":pct(rot_err,95)},"depth_error_m_mean":float(np.mean(depth_err)) if depth_err else None,"lateral_error_m_mean":float(np.mean(lateral_err)) if lateral_err else None},
        "temporal":{"id_switches":id_switches},
        "gate_detection":{"precision":prec,"recall":rec,"f1":f1,"false_positive_gates":fp,"false_negative_gates":fn},
        "runtime":{"mean_inference_ms":float(np.mean(all_times)) if all_times else None,"median_inference_ms":float(np.median(all_times)) if all_times else None,"fps_from_mean_latency":1000.0/float(np.mean(all_times)) if all_times and np.mean(all_times)>0 else None}
      }
    }
    (out/"evaluation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    if rows:
        with (out/"per_sample_metrics.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    return result
