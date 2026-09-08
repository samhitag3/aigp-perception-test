from __future__ import annotations
import argparse,json
from pathlib import Path
import cv2,numpy as np
from gatepose.utils.camera import fov_from_intrinsics

p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); a=p.parse_args(); root=Path(a.dataset)
meta=json.loads((root/'dataset.json').read_text()); errors=[]; warnings=[]
sets={}
for name,file in [('train','train_sequences.txt'),('validation','validation_sequences.txt'),('test','test_sequences.txt')]:
    sets[name]=set(x.strip() for x in (root/'splits'/file).read_text().splitlines() if x.strip())
for x,y in [('train','validation'),('train','test'),('validation','test')]:
    overlap=sets[x]&sets[y]
    if overlap: errors.append(f'split leakage {x}/{y}: {sorted(overlap)[:10]}')

null_v=0; null_w=0; bad_v=0; bad_w=0; missing_pose_for_fallback=0

def valid_vec3(v):
    if v is None: return False
    try: a=np.asarray(v,dtype=np.float32)
    except Exception: return False
    return a.shape==(3,) and bool(np.all(np.isfinite(a)))

def valid_mat4(v):
    if v is None: return False
    try: a=np.asarray(v,dtype=np.float32)
    except Exception: return False
    return a.shape==(4,4) and bool(np.all(np.isfinite(a)))

for sid in sorted(set().union(*sets.values())):
    sd=root/'sequences'/sid
    if not (sd/'frames.jsonl').exists(): errors.append(f'{sid}: missing frames.jsonl'); continue
    prev=-1
    for li,line in enumerate((sd/'frames.jsonl').read_text().splitlines(),1):
        if not line.strip(): continue
        r=json.loads(line); fi=int(r['frame_index']); ts=int(r['timestamp_ns'])
        if ts<=prev: errors.append(f'{sid}:{fi} non-increasing timestamp')
        prev=ts
        rgb=sd/r['files']['rgb']; m=sd/r['files']['instance_mask']
        if not rgb.exists() or not m.exists(): errors.append(f'{sid}:{fi} missing rgb/mask'); continue
        im=cv2.imread(str(rgb)); mm=cv2.imread(str(m),cv2.IMREAD_UNCHANGED)
        if im is None or mm is None or im.shape[:2]!=mm.shape[:2]: errors.append(f'{sid}:{fi} bad dimensions')
        if mm is not None:
            ids={int(x) for x in np.unique(mm) if x!=0}; ann={int(g['mask_id']) for g in r.get('gates',[]) if g.get('mask_id') is not None}
            if ids-ann: errors.append(f'{sid}:{fi} unknown mask ids {ids-ann}')
        for g in r.get('gates',[]):
            kp=g.get('keypoints_2d',{})
            if len(kp)!=8: errors.append(f'{sid}:{fi}:{g.get("track_id")} expected 8 keypoints')

        drone=r.get('drone',{}) or {}
        rv=drone.get('linear_velocity_body_mps')
        rw=drone.get('angular_velocity_body_radps')
        if rv is None: null_v+=1
        elif not valid_vec3(rv): bad_v+=1
        if rw is None: null_w+=1
        elif not valid_vec3(rw): bad_w+=1
        if (not valid_vec3(rv) or not valid_vec3(rw)) and not valid_mat4((r.get('camera',{}) or {}).get('T_world_camera')):
            missing_pose_for_fallback+=1

if null_v: warnings.append(f'{null_v} frames have null linear_velocity_body_mps; loader will derive from camera pose when possible, else use zero')
if null_w: warnings.append(f'{null_w} frames have null angular_velocity_body_radps; loader will derive from camera pose when possible, else use zero')
if bad_v: warnings.append(f'{bad_v} frames have malformed/non-finite linear_velocity_body_mps')
if bad_w: warnings.append(f'{bad_w} frames have malformed/non-finite angular_velocity_body_radps')
if missing_pose_for_fallback: warnings.append(f'{missing_pose_for_fallback} frames lack usable ego vectors and T_world_camera; loader will use zero ego for missing components')

print(f'validated sequences={sum(len(v) for v in sets.values())} errors={len(errors)} warnings={len(warnings)}')
for e in errors[:100]: print('ERROR',e)
for w in warnings: print('WARNING',w)
# AIGP numeric camera sanity
hf,vf=fov_from_intrinsics(640,360,320,320); print(f'AIGP intrinsics imply HFoV={hf:.3f} deg, VFoV={vf:.3f} deg')
if errors: raise SystemExit(2)
