#!/usr/bin/env python3
from __future__ import annotations
import argparse, time
from pathlib import Path
import cv2, numpy as np, torch
from gateperceiver.models import GatePerceiver
from gateperceiver.inference import decode_frame, OnlineTracker, PredictionWriter
from gateperceiver.config import load_yaml
from gateperceiver.contracts import SCHEMA_VERSION


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--checkpoint",required=True); ap.add_argument("--video",required=True); ap.add_argument("--output",required=True); ap.add_argument("--device",default="cuda"); ap.add_argument("--config"); ap.add_argument("--no-tracking",action="store_true"); args=ap.parse_args()
    dev=torch.device(args.device if args.device!="cuda" or torch.cuda.is_available() else "cpu"); ck=torch.load(args.checkpoint,map_location="cpu"); cfg=load_yaml(args.config) if args.config else ck["config"]; model=GatePerceiver(cfg); model.load_state_dict(ck["model"]); model.to(dev).eval()
    cap=cv2.VideoCapture(args.video); fps=float(cap.get(cv2.CAP_PROP_FPS) or 30.); srcW=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); srcH=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); N=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); outH,outW=map(int,cfg["data"].get("image_size",[360,640])); T=int(cfg["data"].get("window_size",1))
    K0=np.array([[320.,0.,320.],[0.,320.,180.],[0.,0.,1.]],np.float32); sx=outW/640.; sy=outH/360.; K0[0]*=sx; K0[1]*=sy
    tracker=None if args.no_tracking else OnlineTracker(); metadata={"schema_name":"uav_gate_perception_predictions","schema_version":SCHEMA_VERSION,"model":{"checkpoint":str(args.checkpoint),"architecture":"GatePerceiver"},"source":{"type":"video","path":str(args.video),"width_px":srcW,"height_px":srcH,"fps":fps,"num_frames":N},"camera":{"assumed_source_intrinsics":{"resolution_px":[640,360],"fx_fy":[320.,320.],"cx_cy":[320.,180.],"hfov_deg":90.0,"vfov_deg":58.7155}},"inference":{"device":str(dev),"temporal_window_size":T,"tracking_enabled":tracker is not None},"outputs_available":{"instance_segmentation":True,"bounding_boxes":True,"keypoints":True,"keypoint_visibility":True,"pose":True,"tracking":tracker is not None}}
    writer=PredictionWriter(args.output,metadata); window=[]
    i=0
    with torch.no_grad():
      while True:
        ok,frame=cap.read()
        if not ok: break
        # If source resolution differs, rescale canonical intrinsics proportionally to source, then to model input.
        Ks=K0.copy(); Ks[0]*=(srcW/640.); Ks[1]*=(srcH/360.); Ks[0]*=(outW/srcW); Ks[1]*=(outH/srcH)
        rgb=cv2.cvtColor(cv2.resize(frame,(outW,outH)),cv2.COLOR_BGR2RGB).astype(np.float32)/255.; ten=torch.from_numpy(rgb).permute(2,0,1); window.append(ten); window=window[-T:]
        while len(window)<T: window.insert(0,window[0])
        inp=torch.stack(window)[None].to(dev); Kin=torch.from_numpy(np.stack([Ks]*T))[None].to(dev)
        if dev.type=="cuda":
            torch.cuda.synchronize()
        t0=time.perf_counter()
        out=model(inp,Kin)
        if dev.type=="cuda":
            torch.cuda.synchronize()
        infer_ms=(time.perf_counter()-t0)*1000
        t1=time.perf_counter(); mask,instances=decode_frame(out,0,T-1,cfg); instances=tracker.update(instances) if tracker else [{**x,"track_id":None, **({} if "track_embedding" not in x else {})} for x in instances]
        for ins in instances: ins.pop("track_embedding",None)
        post_ms=(time.perf_counter()-t1)*1000; writer.write_frame(i,i/fps,mask,instances,{"inference_ms":infer_ms,"postprocess_ms":post_ms,"total_ms":infer_ms+post_ms}); i+=1
    writer.close(); cap.release(); print(f"wrote {i} frames to {args.output}")
if __name__=="__main__": main()
