from __future__ import annotations
import argparse,time,json
from pathlib import Path
import cv2,numpy as np,torch
from gatepose.utils.config import load_config
from gatepose.utils.camera import letterbox_image,transform_K
from gatepose.models import build_model
from gatepose.io.decoder import decode_frame
from gatepose.io.writer import PredictionWriter
p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--video',required=True); p.add_argument('--output',required=True); p.add_argument('--device',default='cuda'); p.add_argument('--fps',type=float); a=p.parse_args(); cfg=load_config(a.config); dev=torch.device(a.device if torch.cuda.is_available() or a.device=='cpu' else 'cpu')
model=build_model(cfg).to(dev); ck=torch.load(a.checkpoint,map_location=dev); model.load_state_dict(ck['model'] if 'model' in ck else ck); model.eval(); cap=cv2.VideoCapture(a.video); sw=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); sh=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps=a.fps or cap.get(cv2.CAP_PROP_FPS) or 30.0; n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); c=cfg['camera']; # scale authoritative AIGP K to source resolution
K=np.array([[c['fx']*sw/c['source_width'],0,c['cx']*sw/c['source_width']],[0,c['fy']*sh/c['source_height'],c['cy']*sh/c['source_height']],[0,0,1]],np.float32)
meta={"schema_name":"uav_gate_perception_predictions","schema_version":"1.0.0","model":{"model_name":"gateposenet_canonical_mg","checkpoint":str(a.checkpoint),"run_id":Path(a.output).name,"git_commit":None},"source":{"type":"video","path":str(a.video),"width_px":sw,"height_px":sh,"fps":fps,"num_frames":n},"inference":{"device":str(dev),"precision":"fp16" if dev.type=='cuda' else 'fp32',"temporal_window_size":cfg['training']['window_size'],"object_confidence_threshold":cfg['decoder']['object_confidence_threshold'],"mask_threshold":cfg['decoder']['mask_threshold'],"tracking_enabled":True},"outputs_available":{"instance_segmentation":True,"union_segmentation":False,"bounding_boxes":True,"keypoints":True,"keypoint_visibility":True,"pose":True,"tracking":True}}
wr=PredictionWriter(a.output,meta); h=None; idx=0; prev=time.perf_counter()
while True:
    ok,bgr=cap.read();
    if not ok: break
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); canvas,tr=letterbox_image(rgb,c['model_width'],c['model_height']); Km=transform_K(K,tr); kin=np.array([Km[0,0]/c['model_width'],Km[1,1]/c['model_height'],Km[0,2]/(c['model_width']-1),Km[1,2]/(c['model_height']-1)],np.float32); image=torch.from_numpy(canvas).permute(2,0,1).float()[None].to(dev)/255; ego=torch.tensor([[0,0,0,0,0,0,1/fps]],dtype=torch.float32,device=dev); ki=torch.from_numpy(kin)[None].to(dev); t0=time.perf_counter()
    with torch.no_grad(): o,h=model.step(image,ego,ki,h)
    infer=(time.perf_counter()-t0)*1000; t1=time.perf_counter(); frame_raw={"instances":{k:v[0] for k,v in o['instances'].items() if k!='pose'}}; frame_raw['instances']['pose']={k:v[0] for k,v in o['instances']['pose'].items()}; mask,inst=decode_frame(frame_raw,tr,(sw,sh),cfg); post=(time.perf_counter()-t1)*1000; wr.write(idx,idx/fps,mask,inst,{"inference_ms":infer,"postprocess_ms":post,"total_ms":infer+post}); idx+=1
wr.close(); cap.release(); print(a.output)
