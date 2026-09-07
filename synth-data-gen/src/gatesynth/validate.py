from __future__ import annotations
from pathlib import Path
import json
import cv2
import numpy as np

REQ_KP=["outer_tl","outer_tr","outer_br","outer_bl","inner_tl","inner_tr","inner_br","inner_bl"]


def validate_dataset(root: str | Path) -> dict:
    root=Path(root)
    errors=[]; warnings=[]; nframes=0; nseq=0
    splits={}
    for s in ("train","validation","test"):
        p=root/"splits"/f"{s}_sequences.txt"
        if not p.exists(): errors.append(f"missing split file {p}"); splits[s]=set()
        else: splits[s]=set(x.strip() for x in p.read_text().splitlines() if x.strip())
    if splits["train"] & splits["validation"] or splits["train"] & splits["test"] or splits["validation"] & splits["test"]:
        errors.append("sequence split overlap detected")
    for seqdir in sorted((root/"sequences").glob("seq_*")):
        nseq+=1
        fp=seqdir/"frames.jsonl"
        if not fp.exists(): errors.append(f"missing {fp}"); continue
        last_idx=-1; last_ts=-1
        with open(fp,"r",encoding="utf-8") as f:
            for ln,line in enumerate(f,1):
                if not line.strip(): continue
                r=json.loads(line); nframes+=1
                if r["frame_index"]<=last_idx: errors.append(f"{fp}:{ln} non-increasing frame_index")
                if r["timestamp_ns"]<=last_ts and last_ts>=0: errors.append(f"{fp}:{ln} non-increasing timestamp")
                last_idx=r["frame_index"]; last_ts=r["timestamp_ns"]
                rgbp=seqdir/r["files"]["rgb"]; mp=seqdir/r["files"]["instance_mask"]
                if not rgbp.exists(): errors.append(f"missing {rgbp}"); continue
                if not mp.exists(): errors.append(f"missing {mp}"); continue
                rgb=cv2.imread(str(rgbp)); mask=cv2.imread(str(mp),cv2.IMREAD_UNCHANGED)
                if rgb is None or mask is None: errors.append(f"unreadable files for {r['frame_id']}"); continue
                if rgb.shape[:2] != mask.shape[:2]: errors.append(f"shape mismatch {r['frame_id']}")
                ids=set(int(x) for x in np.unique(mask) if int(x)!=0)
                anno=set(int(g["mask_id"]) for g in r["gates"] if g["mask_id"] is not None)
                if not ids.issubset(anno): errors.append(f"unknown mask id(s) {ids-anno} in {r['frame_id']}")
                for g in r["gates"]:
                    if set(g["keypoints_2d"].keys()) != set(REQ_KP): errors.append(f"bad keypoint keys {r['frame_id']} {g['track_id']}")
                    T=np.asarray(g["pose"]["T_camera_gate"],float)
                    if T.shape!=(4,4) or not np.allclose(T[3],[0,0,0,1],atol=1e-6): errors.append(f"bad transform {r['frame_id']} {g['track_id']}")
                    if g["mask_id"] is not None:
                        area=int(np.sum(mask==g["mask_id"]))
                        if area != int(g["visibility"]["visible_pixel_area"]): errors.append(f"visible area mismatch {r['frame_id']} {g['track_id']}")
    return {"valid":len(errors)==0,"sequences":nseq,"frames":nframes,"errors":errors[:100],"warnings":warnings}
