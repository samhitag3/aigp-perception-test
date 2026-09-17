#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
from PIL import Image
from gateperceiver.contracts import KEYPOINT_NAMES
from gateperceiver.utils.io import read_json
from gateperceiver.geometry import fov_from_intrinsics


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset-root",required=True); ap.add_argument("--max-frames",type=int); args=ap.parse_args(); root=Path(args.dataset_root)
    meta=read_json(root/"dataset.json"); assert meta["schema_version"].startswith("1.")
    split_sets={}
    for split,file in [("train","train_sequences.txt"),("validation","validation_sequences.txt"),("test","test_sequences.txt")]: split_sets[split]={x.strip() for x in (root/"splits"/file).read_text().splitlines() if x.strip()}
    assert not (split_sets["train"]&split_sets["validation"] or split_sets["train"]&split_sets["test"] or split_sets["validation"]&split_sets["test"]),"sequence leakage across splits"
    checked=0
    for split,seqs in split_sets.items():
      for sid in seqs:
        sd=root/"sequences"/sid; sm=read_json(sd/"sequence.json")
        K=sm["camera"]["intrinsics"]; W=sm["camera"]["width_px"]; H=sm["camera"]["height_px"]
        hf,vf=fov_from_intrinsics(W,H,K["fx"],K["fy"])
        for line in (sd/"frames.jsonl").read_text().splitlines():
            if not line.strip(): continue
            f=json.loads(line); rgbp=sd/f["files"]["rgb"]; mp=sd/f["files"]["instance_mask"]; assert rgbp.exists() and mp.exists()
            rgb=Image.open(rgbp); mask=np.array(Image.open(mp)); assert rgb.size==(mask.shape[1],mask.shape[0])
            declared={int(g["mask_id"]) for g in f.get("gates",[]) if g.get("mask_id") is not None}; present=set(np.unique(mask).tolist())-{0}; assert present<=declared,f"unknown mask IDs in {sid}/{f['frame_index']}: {present-declared}"
            for g in f.get("gates",[]):
                assert set(g.get("keypoints_2d",{}))>=set(KEYPOINT_NAMES)
                T=np.asarray(g["pose"]["T_camera_gate"],dtype=float); assert T.shape==(4,4) and np.all(np.isfinite(T)) and np.allclose(T[3],[0,0,0,1],atol=1e-5)
            checked+=1
            if args.max_frames and checked>=args.max_frames: break
        if args.max_frames and checked>=args.max_frames: break
      if args.max_frames and checked>=args.max_frames: break
    print(f"OK: checked {checked} frames; split leakage absent")
    print(f"first camera-derived FoV: HFoV={hf:.3f} deg, VFoV={vf:.3f} deg")
if __name__=="__main__": main()
