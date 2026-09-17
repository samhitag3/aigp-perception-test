#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from gateperceiver.contracts import KEYPOINT_NAMES
from gateperceiver.utils.io import write_json


def frame_obj(i,W,H):
    x=150+i*3; y=80; ow,oh=260,200; iw,ih=140,100
    pts={"outer_tl":[x,y],"outer_tr":[x+ow,y],"outer_br":[x+ow,y+oh],"outer_bl":[x,y+oh],"inner_tl":[x+60,y+50],"inner_tr":[x+200,y+50],"inner_br":[x+200,y+150],"inner_bl":[x+60,y+150]}
    kps={k:{"projected_px":v,"normalized":[v[0]/(W-1),v[1]/(H-1)],"projection_valid":True,"in_frame":True,"visible":True,"occluded":False,"visibility_state":"visible"} for k,v in pts.items()}
    T=[[1,0,0,0.0],[0,1,0,0.0],[0,0,1,4.0],[0,0,0,1]]
    return {"schema_version":"1.0.0","sequence_id":"","frame_index":i,"sim_time_s":i/30,"files":{"rgb":f"rgb/frame_{i:06d}.jpg","instance_mask":f"instance_masks/frame_{i:06d}.png","depth_mm":None},"image":{"width_px":W,"height_px":H},"camera":{"intrinsics":{"fx":320.,"fy":320.,"cx":320.,"cy":180.,"K":[[320,0,320],[0,320,180],[0,0,1]]}},"gates":[{"track_id":"gate_0001","mask_id":1,"gate_type_id":"standard_gate","pose":{"T_camera_gate":T},"bounding_boxes":{"visible_xyxy_px":[x,y,x+ow,y+oh],"amodal_xyxy_px":[x,y,x+ow,y+oh]},"keypoints_2d":kps}]}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",required=True); args=ap.parse_args(); root=Path(args.output); W,H=640,360
    write_json({"schema_name":"uav_gate_perception_dataset","schema_version":"1.0.0","dataset_id":"tiny_fixture"},root/"dataset.json")
    write_json({"schema_version":"1.0.0","gate_types":{"standard_gate":{"outer_width_m":2.7,"outer_height_m":2.7,"inner_width_m":1.5,"inner_height_m":1.5,"depth_m":0.26}}},root/"gate_geometry.json")
    splits={"train":["seq_train"],"validation":["seq_val"],"test":["seq_test"]}; (root/"splits").mkdir(parents=True,exist_ok=True)
    for s,ids in splits.items(): (root/"splits"/("validation_sequences.txt" if s=="validation" else f"{s}_sequences.txt")).write_text("\n".join(ids)+"\n")
    for sid in ["seq_train","seq_val","seq_test"]:
        sd=root/"sequences"/sid; (sd/"rgb").mkdir(parents=True,exist_ok=True); (sd/"instance_masks").mkdir(exist_ok=True)
        write_json({"schema_version":"1.0.0","sequence_id":sid,"split":"train" if sid=="seq_train" else ("validation" if sid=="seq_val" else "test"),"camera":{"camera_id":"front_camera","width_px":W,"height_px":H,"model":"pinhole","intrinsics":{"fx":320.,"fy":320.,"cx":320.,"cy":180.,"K":[[320,0,320],[0,320,180],[0,0,1]]}},"course":{"gate_tracks":[{"track_id":"gate_0001","gate_type_id":"standard_gate"}]}},sd/"sequence.json")
        lines=[]
        for i in range(6):
            fo=frame_obj(i,W,H); fo["sequence_id"]=sid; x=150+i*3; y=80
            arr=np.zeros((H,W,3),np.uint8); arr[:]=35; arr[y:y+201,x:x+261]=[180,70,30]; Image.fromarray(arr).save(sd/fo["files"]["rgb"],quality=95)
            mask=np.zeros((H,W),np.uint16); mask[y:y+201,x:x+261]=1; mask[y+50:y+151,x+60:x+201]=0; Image.fromarray(mask).save(sd/fo["files"]["instance_mask"])
            lines.append(json.dumps(fo,separators=(",",":")))
        (sd/"frames.jsonl").write_text("\n".join(lines)+"\n")
    print(root)
if __name__=="__main__": main()
