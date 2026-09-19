from __future__ import annotations
import cv2
import numpy as np
from .geometry import (
    KEYPOINT_ORDER, gate_keypoints_local, invert_T, transform_points,
    project_points, polygon_area, rotation_error_components_deg, DEPTH,
    OUTER_W, OUTER_H, INNER_W, INNER_H,
)
from .texture import warp_gate_rgba


def procedural_background(rng: np.random.Generator, w: int, h: int, frame_idx: int, nframes: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h,0:w].astype(np.float32)
    t = frame_idx / max(nframes-1,1)
    base = np.zeros((h,w,3), np.float32)
    sky = 155 + 35*np.sin(0.004*xx + 0.7*t)
    ground = 85 + 25*np.sin(0.006*xx + 0.01*yy + 1.7*t)
    horizon = int(h*(0.52 + 0.03*np.sin(2*np.pi*t)))
    base[:horizon,:,0] = sky[:horizon]
    base[:horizon,:,1] = sky[:horizon] + 15
    base[:horizon,:,2] = sky[:horizon] + 25
    base[horizon:,:,0] = ground[horizon:] + 5
    base[horizon:,:,1] = ground[horizon:] + 20
    base[horizon:,:,2] = ground[horizon:]
    # simple scene clutter, deterministic from passed rng state
    for _ in range(18):
        x = int(rng.integers(0,w)); y = int(rng.integers(horizon,h));
        rw = int(rng.integers(8,70)); rh = int(rng.integers(8,80));
        c = tuple(int(v) for v in rng.integers(45,155,size=3))
        cv2.rectangle(base, (x,y), (min(w-1,x+rw), min(h-1,y+rh)), c, -1)
    return np.clip(base,0,255).astype(np.uint8)


def load_background_frame(background_rgb, w, h, frame_idx, nframes):
    if background_rgb is None:
        return None
    bh,bw = background_rgb.shape[:2]
    scale = max(w/bw, h/bh) * 1.15
    nw,nh = int(round(bw*scale)), int(round(bh*scale))
    im = cv2.resize(background_rgb, (nw,nh), interpolation=cv2.INTER_LINEAR)
    maxx,maxy=max(0,nw-w),max(0,nh-h)
    t=frame_idx/max(nframes-1,1)
    x=int((0.5+0.35*np.sin(2*np.pi*t))*maxx) if maxx else 0
    y=int((0.5+0.25*np.cos(1.3*np.pi*t))*maxy) if maxy else 0
    return im[y:y+h,x:x+w].copy()


def _rgba_over(dst_rgb, src_rgba, instance_mask, mask_id):
    a = src_rgba[...,3].astype(np.float32)/255.0
    m = a > 1e-3
    if not np.any(m):
        return
    dst_rgb[m] = (src_rgba[m,:3].astype(np.float32)*a[m,None] + dst_rgb[m].astype(np.float32)*(1-a[m,None])).astype(np.uint8)
    instance_mask[m] = mask_id


def _gate_local_rects():
    zf, zb = -DEPTH/2, DEPTH/2
    ow,oh=OUTER_W/2,OUTER_H/2; iw,ih=INNER_W/2,INNER_H/2
    outer = np.array([[-ow,-oh,zf],[ow,-oh,zf],[ow,oh,zf],[-ow,oh,zf]],float)
    outer_b = outer.copy(); outer_b[:,2]=zb
    inner = np.array([[-iw,-ih,zf],[iw,-ih,zf],[iw,ih,zf],[-iw,ih,zf]],float)
    inner_b = inner.copy(); inner_b[:,2]=zb
    return outer,outer_b,inner,inner_b


def _side_faces_local():
    o,ob,i,ib=_gate_local_rects()
    faces=[]
    for poly,name in [
        (np.array([o[0],o[1],ob[1],ob[0]]),"outer_top"),
        (np.array([o[1],o[2],ob[2],ob[1]]),"outer_right"),
        (np.array([o[2],o[3],ob[3],ob[2]]),"outer_bottom"),
        (np.array([o[3],o[0],ob[0],ob[3]]),"outer_left"),
        (np.array([i[0],ib[0],ib[1],i[1]]),"inner_top"),
        (np.array([i[1],ib[1],ib[2],i[2]]),"inner_right"),
        (np.array([i[2],ib[2],ib[3],i[3]]),"inner_bottom"),
        (np.array([i[3],ib[3],ib[0],i[0]]),"inner_left"),
    ]:
        faces.append((name,poly))
    return faces


def render_frame(bg_rgb: np.ndarray, gate_skin_rgba: np.ndarray, gates: list[dict], T_world_camera: np.ndarray,
                 K: np.ndarray, width: int, height: int, include_side_faces=True, side_brightness=0.55):
    rgb = bg_rgb.copy()
    inst = np.zeros((height,width), np.uint16)
    T_camera_world = invert_T(T_world_camera)
    kplocal = gate_keypoints_local()
    all_local = np.stack([kplocal[k] for k in KEYPOINT_ORDER],axis=0)

    records=[]
    layers=[]
    candidate=[]
    for g in gates:
        T_cg=T_camera_world @ g["T_world_gate"]
        pts_cam=transform_points(T_cg, all_local)
        uv, valid=project_points(K,pts_cam)
        outer_uv=uv[:4]; inner_uv=uv[4:]
        center=T_cg[:3,3]
        if center[2] <= 0.15:
            continue
        # keep gates that are in or near FOV, including partials
        if np.all(~valid[:4]):
            continue
        if np.nanmax(outer_uv[:,0]) < -width*0.35 or np.nanmin(outer_uv[:,0]) > width*1.35 or np.nanmax(outer_uv[:,1]) < -height*0.35 or np.nanmin(outer_uv[:,1]) > height*1.35:
            continue
        candidate.append((g,T_cg,pts_cam,uv,valid))

    for local_id,(g,T_cg,pts_cam,uv,valid) in enumerate(candidate,start=1):
        outer_uv=uv[:4]; inner_uv=uv[4:]
        zmean=float(np.mean(pts_cam[:4,2]))
        warped=warp_gate_rgba(gate_skin_rgba, outer_uv, (width,height))
        layers.append((zmean,"front",local_id,warped,None))
        if include_side_faces:
            # use mean gate skin color for side faces
            alpha=gate_skin_rgba[...,3]>0
            mean_color=gate_skin_rgba[...,:3][alpha].mean(axis=0) if np.any(alpha) else np.array([100,100,100])
            side_color=np.clip(mean_color*side_brightness,0,255).astype(np.uint8)
            for name,poly_local in _side_faces_local():
                pc=transform_points(T_cg,poly_local)
                puv,pvalid=project_points(K,pc)
                if not pvalid.all() or not np.isfinite(puv).all():
                    continue
                rgba=np.zeros((height,width,4),np.uint8)
                pts=np.round(puv).astype(np.int32)
                cv2.fillConvexPoly(rgba,pts,tuple(int(v) for v in side_color) + (255,))
                layers.append((float(np.mean(pc[:,2])),"side",local_id,rgba,name))

    # far -> near painter's algorithm
    layers.sort(key=lambda x:x[0], reverse=True)
    for _,_,mask_id,rgba,_ in layers:
        _rgba_over(rgb,rgba,inst,mask_id)

    # build records after final visibility known
    for local_id,(g,T_cg,pts_cam,uv,valid) in enumerate(candidate,start=1):
        outer_uv=uv[:4]; inner_uv=uv[4:]
        amodal=np.zeros((height,width),np.uint8)
        if valid[:4].all() and np.isfinite(outer_uv).all():
            cv2.fillConvexPoly(amodal,np.round(outer_uv).astype(np.int32),1)
        if valid[4:].all() and np.isfinite(inner_uv).all():
            cv2.fillConvexPoly(amodal,np.round(inner_uv).astype(np.int32),0)
        amodal_in=int(amodal.sum())
        full=max(0.0, polygon_area(outer_uv)-polygon_area(inner_uv)) if valid.all() else 0.0
        vismask=(inst==local_id)
        visible=int(vismask.sum())
        occl=1.0-visible/max(amodal_in,1) if amodal_in>0 else 0.0
        trunc=1.0-amodal_in/max(full,1.0) if full>0 else 0.0
        occl=float(np.clip(occl,0,1)); trunc=float(np.clip(trunc,0,1))

        if visible>0:
            ys,xs=np.where(vismask)
            vb=[float(xs.min()),float(ys.min()),float(xs.max()),float(ys.max())]
            final_mask_id=local_id
        else:
            vb=None; final_mask_id=None
        ab=[float(np.nanmin(outer_uv[:,0])),float(np.nanmin(outer_uv[:,1])),float(np.nanmax(outer_uv[:,0])),float(np.nanmax(outer_uv[:,1]))] if valid[:4].all() else None

        kp2d={}
        for j,name in enumerate(KEYPOINT_ORDER):
            p=uv[j]; ok=bool(valid[j] and np.isfinite(p).all())
            in_frame=bool(ok and 0<=p[0]<width and 0<=p[1]<height)
            visible_k=False
            if in_frame and final_mask_id is not None:
                x=int(round(p[0])); y=int(round(p[1]))
                x0,x1=max(0,x-3),min(width,x+4); y0,y1=max(0,y-3),min(height,y+4)
                visible_k=bool(np.any(inst[y0:y1,x0:x1]==local_id))
            if not ok: state="behind_camera"
            elif not in_frame: state="out_of_frame"
            elif visible_k: state="visible"
            else: state="occluded"
            kp2d[name]={
                "projected_px": [float(p[0]),float(p[1])] if ok else None,
                "normalized": [float(p[0]/(width-1)),float(p[1]/(height-1))] if ok else None,
                "projection_valid": ok,
                "in_frame": in_frame,
                "visible": visible_k,
                "occluded": bool(state=="occluded"),
                "visibility_state": state,
            }

        center=T_cg[:3,3]
        distance=float(np.linalg.norm(center))
        kp3cam={name:[float(v) for v in pts_cam[j]] for j,name in enumerate(KEYPOINT_ORDER)}
        worldpts=transform_points(g["T_world_gate"],all_local)
        kp3world={name:[float(v) for v in worldpts[j]] for j,name in enumerate(KEYPOINT_ORDER)}
        records.append({
            "track_id": g["track_id"],
            "mask_id": final_mask_id,
            "gate_type_id": g["gate_type_id"],
            "route_order_index": g["route_order_index"],
            "pose": {
                "T_world_gate": g["T_world_gate"].tolist(),
                "T_camera_gate": T_cg.tolist(),
                "center_camera_m": [float(v) for v in center],
                "distance_camera_m": distance,
                "depth_camera_z_m": float(center[2]),
                "relative_rotation_deg": rotation_error_components_deg(T_cg),
                "view_angle_deg": float(np.degrees(np.arctan2(np.linalg.norm(center[:2]),max(center[2],1e-9)))),
            },
            "visibility": {
                "visible_pixel_area": visible,
                "amodal_area_in_frame_px": amodal_in,
                "amodal_area_full_px": float(full),
                "visible_fraction": float(visible/max(amodal_in,1)) if amodal_in>0 else 0.0,
                "occlusion_fraction": occl,
                "truncation_fraction": trunc,
                "has_visible_pixels": bool(visible>0),
                "fully_visible": bool(occl<0.02 and trunc<0.02),
                "partially_occluded": bool(occl>=0.02),
                "partially_out_of_frame": bool(trunc>=0.02),
            },
            "bounding_boxes": {"visible_xyxy_px": vb, "amodal_xyxy_px": ab},
            "keypoints_2d": kp2d,
            "keypoints_3d_camera_m": kp3cam,
            "keypoints_3d_world_m": kp3world,
        })
    return rgb,inst,records
