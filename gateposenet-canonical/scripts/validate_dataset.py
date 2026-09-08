from __future__ import annotations
import argparse,json,math
from pathlib import Path
import cv2,numpy as np
from gatepose.utils.camera import fov_from_intrinsics
p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); a=p.parse_args(); root=Path(a.dataset)
meta=json.loads((root/'dataset.json').read_text()); errors=[]; warnings=[]
sets={}
for name,file in [('train','train_sequences.txt'),('validation','validation_sequences.txt'),('test','test_sequences.txt')]: sets[name]=set(x.strip() for x in (root/'splits'/file).read_text().splitlines() if x.strip())
for x,y in [('train','validation'),('train','test'),('validation','test')]:
    overlap=sets[x]&sets[y]
    if overlap: errors.append(f'split leakage {x}/{y}: {sorted(overlap)[:10]}')
for sid in sorted(set().union(*sets.values())):
    sd=root/'sequences'/sid
    if not (sd/'frames.jsonl').exists(): errors.append(f'{sid}: missing frames.jsonl'); continue
    prev=-1
    for li,line in enumerate((sd/'frames.jsonl').read_text().splitlines(),1):
        if not line.strip(): continue
        r=json.loads(line); fi=int(r['frame_index']); ts=int(r['timestamp_ns']);
        if ts<=prev: errors.append(f'{sid}:{fi} non-increasing timestamp'); prev=ts
        rgb=sd/r['files']['rgb']; m=sd/r['files']['instance_mask'];
        if not rgb.exists() or not m.exists(): errors.append(f'{sid}:{fi} missing rgb/mask'); continue
        im=cv2.imread(str(rgb)); mm=cv2.imread(str(m),cv2.IMREAD_UNCHANGED)
        if im is None or mm is None or im.shape[:2]!=mm.shape[:2]: errors.append(f'{sid}:{fi} bad dimensions')
        ids={int(x) for x in np.unique(mm) if x!=0}; ann={int(g['mask_id']) for g in r.get('gates',[]) if g.get('mask_id') is not None}
        if ids-ann: errors.append(f'{sid}:{fi} unknown mask ids {ids-ann}')
        for g in r.get('gates',[]):
            kp=g.get('keypoints_2d',{})
            if len(kp)!=8: errors.append(f'{sid}:{fi}:{g.get("track_id")} expected 8 keypoints')
print(f'validated sequences={sum(len(v) for v in sets.values())} errors={len(errors)} warnings={len(warnings)}')
for e in errors[:100]: print('ERROR',e)
# AIGP numeric camera sanity
hf,vf=fov_from_intrinsics(640,360,320,320); print(f'AIGP intrinsics imply HFoV={hf:.3f} deg, VFoV={vf:.3f} deg')
if errors: raise SystemExit(2)
