from __future__ import annotations
from pathlib import Path
import json
import cv2
import numpy as np


class PredictionWriter:
    def __init__(self,out_dir,metadata):
        self.root=Path(out_dir); (self.root/"instance_masks").mkdir(parents=True,exist_ok=True); (self.root/"overlays").mkdir(parents=True,exist_ok=True)
        (self.root/"inference.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
        self.fp=(self.root/"frames.jsonl").open("w",encoding="utf-8")
    def write(self,frame_index,timestamp_s,mask,instances,runtime):
        rel=f"instance_masks/frame_{frame_index:06d}.png"; cv2.imwrite(str(self.root/rel),mask.astype(np.uint16))
        rec={"frame_index":int(frame_index),"timestamp_s":float(timestamp_s),"instance_mask":rel,"num_gate_instances":len(instances),"instances":instances,"runtime":runtime}
        self.fp.write(json.dumps(rec,separators=(",",":"))+"\n"); self.fp.flush()
    def close(self): self.fp.close()
