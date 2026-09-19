from __future__ import annotations
from pathlib import Path
import json
from PIL import Image
import numpy as np
from gateperceiver.utils.io import write_json

class PredictionWriter:
    def __init__(self, out_dir, metadata):
        self.root=Path(out_dir); (self.root/"instance_masks").mkdir(parents=True,exist_ok=True); self.f=(self.root/"frames.jsonl").open("w",encoding="utf-8"); write_json(metadata,self.root/"inference.json")
    def write_frame(self,frame_index,timestamp_s,mask,instances,runtime):
        rel=f"instance_masks/frame_{frame_index:06d}.png"; Image.fromarray(mask.astype(np.uint16)).save(self.root/rel)
        obj={"frame_index":int(frame_index),"timestamp_s":float(timestamp_s),"instance_mask":rel,"num_gate_instances":len(instances),"instances":instances,"runtime":runtime}
        self.f.write(json.dumps(obj,separators=(",",":"))+"\n"); self.f.flush()
    def close(self): self.f.close()
