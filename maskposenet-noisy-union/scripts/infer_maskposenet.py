#!/usr/bin/env python3
"""MaskPoseNet-MG union-mask inference.

Model input is ONLY the mask sequence. --rgb-input is visualization-only.
Any non-zero input pixel is treated as gate foreground; instance IDs/colors are
deliberately discarded before model inference.
"""
from __future__ import annotations
import argparse, json, math, re, time, subprocess, sys
from pathlib import Path
from typing import Iterator
import cv2, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gatenet.utils import load_checkpoint, load_config, pick_device
from gateposenet.model import build_gateposenet_mg

SUFFIXES={'.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp'}
PALETTE=np.asarray([[230,25,75],[60,180,75],[255,225,25],[0,130,200],[245,130,48],[145,30,180],[70,240,240],[240,50,230],[210,245,60],[250,190,212],[0,128,128],[220,190,255],[170,110,40],[255,250,200],[128,0,0],[170,255,195]],dtype=np.uint8)

def gate_bgr(index: int):
    """Return a normal Python BGR tuple for OpenCV drawing."""
    c = PALETTE[int(index) % len(PALETTE)]
    return (int(c[2]), int(c[1]), int(c[0]))

def alpha_blend_mask(image, mask, bgr, alpha):
    """Blend a solid BGR color into masked pixels without cv2.addWeighted on fancy-indexed arrays."""
    if not np.any(mask):
        return
    a = float(np.clip(alpha, 0.0, 1.0))
    color = np.asarray(bgr, dtype=np.float32)
    src = image[mask].astype(np.float32)
    image[mask] = np.clip(src * (1.0 - a) + color * a, 0, 255).astype(np.uint8)

def nkey(p): return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)',p.name)]
def files_in(p): return sorted([x for x in Path(p).iterdir() if x.suffix.lower() in SUFFIXES], key=nkey)

def color_to_labels(frame):
    if frame.ndim==2: return frame.astype(np.int32)
    if frame.shape[2]==1: return frame[...,0].astype(np.int32)
    # grayscale-like 3-channel frames keep their label values.
    if np.array_equal(frame[...,0],frame[...,1]) and np.array_equal(frame[...,1],frame[...,2]):
        return frame[...,0].astype(np.int32)
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    flat=rgb.reshape(-1,3); uniq=np.unique(flat,axis=0)
    out=np.zeros(rgb.shape[:2],np.int32); k=1
    for c in uniq:
        if int(c.max())<10: continue
        out[np.all(rgb==c,axis=2)]=k; k+=1
    return out

def labels_to_model(labels,h,w):
    # UNION-INPUT VARIANT: discard all frame-local instance IDs/colors.
    # Background stays 0; every detected gate pixel becomes 1 in all 3
    # channels. Keeping three channels preserves the original encoder exactly.
    union=(labels>0).astype(np.uint8)*255
    rgb=np.repeat(union[...,None],3,axis=2)
    r=cv2.resize(rgb,(w,h),interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(r.transpose(2,0,1)).float().unsqueeze(0)/255.0

def white_mask_base(labels):
    u=(labels>0).astype(np.uint8)*210
    return cv2.cvtColor(u,cv2.COLOR_GRAY2BGR)

def iter_source(path):
    p=Path(path)
    if p.is_dir():
        for f in files_in(p):
            im=cv2.imread(str(f),cv2.IMREAD_UNCHANGED)
            if im is None: continue
            yield color_to_labels(im), f.name
    else:
        cap=cv2.VideoCapture(str(p)); i=0
        while True:
            ok,im=cap.read()
            if not ok: break
            yield color_to_labels(im), f'{i:06d}'; i+=1
        cap.release()

def rgb_reader(path):
    if path is None: return None
    p=Path(path)
    if p.is_dir():
        fs=files_in(p)
        def get(i): return cv2.imread(str(fs[i]),cv2.IMREAD_COLOR) if i<len(fs) else None
        return get
    cap=cv2.VideoCapture(str(p))
    def get(i):
        ok,x=cap.read(); return x if ok else None
    get.cap=cap
    return get

def gpu_util(device):
    if device.type!='cuda': return None
    try: return int(torch.cuda.utilization(device))
    except Exception: return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True,help='mask folder/video; all non-zero pixels are unionized before model inference')
    ap.add_argument('--rgb-input',default=None,help='optional RGB folder/video; visualization only')
    ap.add_argument('--checkpoint',required=True); ap.add_argument('--config',required=True); ap.add_argument('--out-dir',required=True)
    ap.add_argument('--device',default='cuda'); ap.add_argument('--fps',type=float,default=60.0)
    ap.add_argument('--presence-th',type=float,default=None,help='defaults to config inference.presence_threshold or 0.5'); ap.add_argument('--mask-th',type=float,default=None,help='defaults to config inference.mask_threshold or 0.5'); ap.add_argument('--overlay-alpha',type=float,default=.35)
    ap.add_argument('--max-frames',type=int,default=None); ap.add_argument('--gpu-sample-every',type=int,default=30)
    args=ap.parse_args()
    cfg=load_config(args.config); device=pick_device(args.device)
    infer_cfg=cfg.get('inference',{})
    presence_th=float(args.presence_th if args.presence_th is not None else infer_cfg.get('presence_threshold',0.5))
    mask_th=float(args.mask_th if args.mask_th is not None else infer_cfg.get('mask_threshold',0.5))
    model=build_gateposenet_mg(cfg['model']).to(device).eval(); load_checkpoint(args.checkpoint,model,map_location=device)
    h=int(cfg['data'].get('height',192)); w=int(cfg['data'].get('width',320))
    out=Path(args.out_dir); (out/'overlays').mkdir(parents=True,exist_ok=True); (out/'pred_masks').mkdir(parents=True,exist_ok=True)
    getrgb=rgb_reader(args.rgb_input)
    hidden=None; rows=[]; gpu_ms=[]; wall_ms=[]; e2e_ms=[]; utils=[]; writer=None
    ego=torch.zeros(1,7,device=device); ego[0,6]=1.0/max(args.fps,1e-6)
    warm=5
    for idx,(labels,name) in enumerate(iter_source(args.input)):
        if args.max_frames is not None and idx>=args.max_frames: break
        t_e2e=time.perf_counter(); H,W=labels.shape[:2]
        x=labels_to_model(labels,h,w).to(device,non_blocking=True)
        if device.type=='cuda': torch.cuda.synchronize(); ev0=torch.cuda.Event(enable_timing=True); ev1=torch.cuda.Event(enable_timing=True); ev0.record()
        t0=time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type='cuda',enabled=(device.type=='cuda')):
            pred,hidden=model.step(x,ego,hidden)
        if device.type=='cuda': ev1.record(); torch.cuda.synchronize(); gm=ev0.elapsed_time(ev1)
        else: gm=(time.perf_counter()-t0)*1000
        wm=(time.perf_counter()-t0)*1000
        pres=torch.sigmoid(pred['presence_logit'])[0].detach().cpu().numpy()
        keep=np.where(pres>=presence_th)[0].tolist()
        masks=torch.sigmoid(pred['mask_logit'][0]).detach().cpu()
        corners=pred['corners_uv'][0].detach().cpu().numpy(); pos=pred['position'][0].detach().cpu().numpy()
        base=(getrgb(idx) if getrgb else None)
        if base is None: base=white_mask_base(labels)
        if base.shape[:2]!=(H,W): base=cv2.resize(base,(W,H))
        overlay=base.copy(); pred_id=np.zeros((H,W),np.uint16); gates=[]
        for j,q in enumerate(keep):
            bgr=gate_bgr(j)
            pm=F.interpolate(masks[q][None,None],size=(H,W),mode='bilinear',align_corners=False)[0,0].numpy()>=mask_th
            pred_id[pm]=j+1
            alpha_blend_mask(overlay, pm, bgr, args.overlay_alpha)
            uv=corners[q]*np.array([W,H],np.float32); pts=np.round(uv).astype(np.int32)
            # black outline + colored square for visibility
            cv2.polylines(overlay,[pts],True,(0,0,0),6,cv2.LINE_AA); cv2.polylines(overlay,[pts],True,bgr,3,cv2.LINE_AA)
            for p in pts: cv2.circle(overlay,tuple(p),7,(0,0,0),-1,cv2.LINE_AA); cv2.circle(overlay,tuple(p),4,bgr,-1,cv2.LINE_AA)
            c=pts.mean(axis=0).astype(int); cv2.putText(overlay,f'G{j+1} {pres[q]:.2f} xyz={pos[q,0]:.2f},{pos[q,1]:.2f},{pos[q,2]:.2f}',tuple(c),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),3,cv2.LINE_AA); cv2.putText(overlay,f'G{j+1} {pres[q]:.2f} xyz={pos[q,0]:.2f},{pos[q,1]:.2f},{pos[q,2]:.2f}',tuple(c),cv2.FONT_HERSHEY_SIMPLEX,.45,bgr,1,cv2.LINE_AA)
            gates.append({'query':int(q),'score':float(pres[q]),'corners_px':uv.tolist(),'position_camera_m':pos[q].tolist()})
        if not getrgb:
            # outline input instances too, so neighboring gates remain distinct on B/W base
            for mid in [int(v) for v in np.unique(labels) if int(v)>0]:
                cnts,_=cv2.findContours((labels==mid).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(overlay,cnts,-1,(0,0,0),2)
        util=gpu_util(device) if idx%max(1,args.gpu_sample_every)==0 else None
        if util is not None: utils.append(util)
        e2e=(time.perf_counter()-t_e2e)*1000
        if idx>=warm: gpu_ms.append(gm); wall_ms.append(wm); e2e_ms.append(e2e)
        cv2.putText(overlay,f'model {gm:.2f} ms ({1000/max(gm,1e-6):.0f} Hz) | e2e {e2e:.2f} ms', (10,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),3,cv2.LINE_AA); cv2.putText(overlay,f'model {gm:.2f} ms ({1000/max(gm,1e-6):.0f} Hz) | e2e {e2e:.2f} ms', (10,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
        cv2.imwrite(str(out/'overlays'/f'frame_{idx:06d}.jpg'),overlay); cv2.imwrite(str(out/'pred_masks'/f'frame_{idx:06d}.png'),pred_id)
        if writer is None: writer=cv2.VideoWriter(str(out/'overlay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),args.fps,(W,H))
        writer.write(overlay)
        rows.append({'frame_index':idx,'source_name':name,'model_gpu_ms':gm,'model_wall_ms':wm,'end_to_end_ms':e2e,'gpu_util_percent_sample':util,'gates':gates})
    if writer: writer.release()
    if getrgb is not None and hasattr(getrgb,'cap'): getrgb.cap.release()
    def stats(a):
        if not a:return {}
        x=np.asarray(a); return {'mean':float(x.mean()),'median':float(np.median(x)),'p95':float(np.percentile(x,95)),'p99':float(np.percentile(x,99)),'hz_from_mean':float(1000/x.mean())}
    summary={'frames':len(rows),'warmup_excluded':min(warm,len(rows)),'model_gpu_ms':stats(gpu_ms),'model_wall_ms':stats(wall_ms),'end_to_end_ms':stats(e2e_ms),'gpu_util_percent_mean_sampled':float(np.mean(utils)) if utils else None,'cuda_memory_allocated_mb':float(torch.cuda.max_memory_allocated()/2**20) if device.type=='cuda' else None,'cuda_memory_reserved_mb':float(torch.cuda.max_memory_reserved()/2**20) if device.type=='cuda' else None,'target_90hz_budget_ms':1000/90,'presence_threshold':presence_th,'mask_threshold':mask_th}
    (out/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows)); (out/'inference.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
