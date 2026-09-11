from __future__ import annotations
import argparse
from collections import deque
from pathlib import Path
import json
import time

import cv2
import numpy as np
import torch
from PIL import Image

from gateperception.data.canonical import CanonicalIndex, _load_rgb
from gateperception.geometry.camera import camera_matrix
from gateperception.geometry.pnp import solve_gate_pose
from gateperception.inference.decoder import decode_segmentation_frame
from gateperception.inference.tracker import GateTracker
from gateperception.inference.keypoints import KeypointTrackMemory
from gateperception.models import build_segmentation_model, build_keypoint_model
from gateperception.utils.config import load_yaml


def load_model(path, builder, device):
    ck = torch.load(path, map_location="cpu")
    cfg = ck["config"]
    m = builder(cfg).to(device); m.load_state_dict(ck["model"], strict=True); m.eval()
    return m, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--seg-checkpoint", required=True)
    ap.add_argument("--keypoint-checkpoint", required=True)
    ap.add_argument("--camera-config", default="configs/camera.yaml")
    ap.add_argument("--gate-geometry", default="configs/gate_geometry.yaml")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--object-threshold", type=float, default=0.5)
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    args = ap.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    seg, scfg = load_model(args.seg_checkpoint, build_segmentation_model, device)
    kp, kcfg = load_model(args.keypoint_checkpoint, build_keypoint_model, device)
    cam = load_yaml(args.camera_config)["camera"]
    geom = load_yaml(args.gate_geometry)["gate"]
    W, H = int(cam["width_px"]), int(cam["height_px"])
    K = camera_matrix(cam["fx"], cam["fy"], cam["cx"], cam["cy"])
    distortion = np.asarray(cam.get("distortion_coefficients", [0,0,0,0,0]), dtype=np.float64)
    index = CanonicalIndex(args.dataset, [args.split])
    out_root = Path(args.output); out_root.mkdir(parents=True, exist_ok=True)
    seg_window = int(scfg["model"].get("window_size", 5))
    for sid in index.sequence_ids:
        seq_out = out_root / sid; (seq_out / "instance_masks").mkdir(parents=True, exist_ok=True)
        tracker = GateTracker(max_age=8)
        kp_mem = KeypointTrackMemory(kp, device, int(kcfg["model"].get("window_size", 7)), tuple(kcfg["model"].get("crop_size", [192,192])), float(kcfg["model"].get("crop_padding", 0.35)))
        hist = deque(maxlen=seg_window)
        with open(seq_out / "frames.jsonl", "w", encoding="utf-8") as fout:
            for row in index.sequences[sid]:
                rgb_src = _load_rgb(index.frame_path(sid, row["files"]["rgb"]))
                if rgb_src.shape[:2] != (H, W):
                    rgb_np = cv2.resize(rgb_src, (W, H), interpolation=cv2.INTER_AREA)
                else:
                    rgb_np = rgb_src
                rgb_t = torch.from_numpy(rgb_np.copy()).permute(2,0,1).float()/255.0
                hist.append(rgb_t); frames=list(hist)
                while len(frames) < seg_window: frames.insert(0, frames[0])
                x=torch.stack(frames)[None].to(device)
                t0=time.perf_counter()
                with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type=="cuda"):
                    raw=seg(x, return_all_frames=False)["frames"][-1]
                raw0={k:v[0] for k,v in raw.items()}
                inst,dets=decode_segmentation_frame(raw0,H,W,args.object_threshold,args.mask_threshold)
                dets=tracker.update(int(row["frame_index"]),dets)
                dets=dets+tracker.memory_only(int(row["frame_index"]),max_emit_age=2)
                t1=time.perf_counter()
                instances=[]
                for det in dets:
                    kps=kp_mem.predict(rgb_np,det)
                    pose=solve_gate_pose(kps,geom["keypoints_gate_frame_m"],K,distortion)
                    instances.append({"mask_id":det.mask_id,"source":det.source,"track_id":det.track_id,"detection_score":det.score,"mask_score":None if det.mask_id is None else det.score,"box_xyxy_px":det.box_xyxy_px,"visible_area_px":int(det.mask.sum()),"keypoints":kps,"pose":pose})
                t2=time.perf_counter()
                name=f"frame_{int(row['frame_index']):06d}.png"
                Image.fromarray(inst, mode="I;16").save(seq_out/"instance_masks"/name)
                pred={"sequence_id":sid,"frame_index":int(row["frame_index"]),"timestamp_s":row.get("sim_time_s"),"instance_mask":f"instance_masks/{name}","num_gate_instances":len(instances),"num_mask_instances":sum(1 for x in instances if x["mask_id"] is not None),"instances":instances,"runtime":{"inference_ms":(t1-t0)*1000,"postprocess_and_downstream_ms":(t2-t1)*1000,"total_ms":(t2-t0)*1000}}
                fout.write(json.dumps(pred)+"\n")
        print("finished", sid)

if __name__ == "__main__":
    main()
