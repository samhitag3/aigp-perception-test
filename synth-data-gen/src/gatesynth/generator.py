from __future__ import annotations
from pathlib import Path
import json
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from .augment import sample_sequence_noise, apply_photometric
from .geometry import invert_T, DEPTH, OUTER_W, OUTER_H, INNER_W, INNER_H
from .texture import load_gate_skin
from .trajectory import generate_course, generate_camera_trajectory
from .render import render_frame, procedural_background, load_background_frame
from .writer import ensure_sequence_dirs, save_rgb, save_mask, json_dump, write_jsonl, write_checksums


def _choose_background(rng, cfg, width, height):
    d=cfg["render"].get("backgrounds_dir")
    if d:
        paths=[]
        for ext in ("*.jpg","*.jpeg","*.png"):
            paths += list(Path(d).glob(ext))
        if paths and rng.random() > cfg["render"].get("procedural_background_probability",0.5):
            p=paths[int(rng.integers(0,len(paths)))]
            bgr=cv2.imread(str(p),cv2.IMREAD_COLOR)
            if bgr is not None:
                return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),str(p)
    return None,"procedural"


def _dataset_json(cfg):
    return {
      "schema_name":"uav_gate_perception_dataset","schema_version":cfg["schema_version"],
      "dataset_id":cfg["dataset_id"],
      "coordinate_systems":{
        "world":{"handedness":"right","x_axis":"forward_reference","y_axis":"left","z_axis":"up","units":"meters"},
        "drone_body":{"convention":"FLU","x_axis":"forward","y_axis":"left","z_axis":"up","handedness":"right"},
        "camera_optical":{"x_axis":"right","y_axis":"down","z_axis":"forward","handedness":"right"},
        "gate_local":{"x_axis":"right_when_viewed_from_approach_side","y_axis":"down_when_viewed_from_approach_side","z_axis":"through_gate_from_approach_side","handedness":"right"}
      },
      "transform_convention":{"name":"T_A_B","definition":"T_A_B transforms a homogeneous point expressed in coordinate frame B into coordinate frame A","matrix_layout":"row_major","quaternion_order":"xyzw","translation_units":"meters"},
      "image_coordinate_system":{"origin":"top_left","x_direction":"right","y_direction":"down","pixel_coordinate_type":"float","top_left_pixel_center":[0.0,0.0],"in_frame_rule":"0 <= x < image_width and 0 <= y < image_height","normalized_x":"x / (image_width - 1)","normalized_y":"y / (image_height - 1)"},
      "gate_keypoint_order":["outer_tl","outer_tr","outer_br","outer_bl","inner_tl","inner_tr","inner_br","inner_bl"],
      "mask_encoding":{"format":"PNG","dtype":"uint16","channels":1,"background_value":0,"meaning":"Each nonzero integer is a frame-local gate mask_id"},
      "rgb_encoding":{"format":"JPEG","color_space":"sRGB","channels":3,"dtype":"uint8"},
      "depth_encoding":{"format":"PNG","dtype":"uint16","units":"millimeters","invalid_value":0,"meaning":"camera optical Z depth, not Euclidean range"},
      "required_modalities":{"rgb":True,"instance_mask":True,"frame_annotations":True,"camera_intrinsics":True,"camera_pose":True,"drone_pose":True,"gate_pose":True,"gate_geometry":True,"projected_keypoints":True,"keypoint_visibility":True},
      "optional_modalities":{"dense_depth":False,"optical_flow":False,"surface_normals":False,"imu":False,"control_actions":False},
      "splitting":{"unit":"sequence","train_fraction":cfg["split"]["train"],"validation_fraction":cfg["split"]["validation"],"test_fraction":cfg["split"]["test"],"frames_from_same_sequence_may_cross_splits":False}
    }


def _gate_geometry_json(cfg):
    z=-DEPTH/2
    return {"schema_version":cfg["schema_version"],"gate_types":{"standard_gate":{
      "outer_width_m":OUTER_W,"outer_height_m":OUTER_H,"inner_width_m":INNER_W,"inner_height_m":INNER_H,"depth_m":DEPTH,"keypoint_reference_surface":"approach_face",
      "keypoints_gate_frame_m":{
        "outer_tl":[-OUTER_W/2,-OUTER_H/2,z],"outer_tr":[OUTER_W/2,-OUTER_H/2,z],"outer_br":[OUTER_W/2,OUTER_H/2,z],"outer_bl":[-OUTER_W/2,OUTER_H/2,z],
        "inner_tl":[-INNER_W/2,-INNER_H/2,z],"inner_tr":[INNER_W/2,-INNER_H/2,z],"inner_br":[INNER_W/2,INNER_H/2,z],"inner_bl":[-INNER_W/2,INNER_H/2,z]
      }}}}


def generate_dataset(cfg: dict, gate_skin_path: str, output_root: str | None=None):
    root=Path(output_root or cfg["output_root"]).expanduser().resolve()
    root.mkdir(parents=True,exist_ok=True); (root/"splits").mkdir(exist_ok=True)
    skin=load_gate_skin(gate_skin_path)
    json_dump(root/"dataset.json",_dataset_json(cfg)); json_dump(root/"gate_geometry.json",_gate_geometry_json(cfg))

    nseq=int(cfg["sequences"]["count"]); base_seed=int(cfg["seed"])
    seq_ids=[f"seq_{i:06d}" for i in range(nseq)]
    split_rng=np.random.default_rng(base_seed+99173); shuffled=seq_ids.copy(); split_rng.shuffle(shuffled)
    ntr=int(round(nseq*cfg["split"]["train"])); nv=int(round(nseq*cfg["split"]["validation"]))
    split_map={s:"train" for s in shuffled[:ntr]}; split_map.update({s:"validation" for s in shuffled[ntr:ntr+nv]}); split_map.update({s:"test" for s in shuffled[ntr+nv:]})
    for name in ("train","validation","test"):
        ids=[s for s in seq_ids if split_map[s]==name]
        (root/"splits"/f"{name}_sequences.txt").write_text("\n".join(ids)+"\n",encoding="utf-8")

    W,H=int(cfg["image"]["width"]),int(cfg["image"]["height"]); fps=float(cfg["sequences"]["fps"])
    for si,seq_id in enumerate(tqdm(seq_ids,desc="Sequences")):
        rng=np.random.default_rng(base_seed + si*100003)
        frames=int(rng.integers(cfg["sequences"]["frames_min"],cfg["sequences"]["frames_max"]+1))
        gates=generate_course(rng,cfg["sequences"]); traj=generate_camera_trajectory(rng,gates,frames,cfg["sequences"])
        noise=sample_sequence_noise(rng,cfg["noise"]); bg,bg_id=_choose_background(rng,cfg,W,H)
        fx=float(rng.uniform(*cfg["camera"]["fx_range"])); fy=fx*float(rng.uniform(*cfg["camera"]["fy_scale_range"]));
        cx=W/2+float(rng.uniform(-cfg["camera"]["principal_point_jitter_px"],cfg["camera"]["principal_point_jitter_px"])); cy=H/2+float(rng.uniform(-cfg["camera"]["principal_point_jitter_px"],cfg["camera"]["principal_point_jitter_px"]));
        K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],float)
        seqdir=ensure_sequence_dirs(root,seq_id)
        seq_meta={
          "schema_version":cfg["schema_version"],"sequence_id":seq_id,"split":split_map[seq_id],"num_frames":frames,
          "simulator":{"name":"synthetic_projective_renderer","version":"0.1.0","simulation_seed":int(base_seed+si*100003)},
          "timing":{"nominal_fps":fps,"nominal_dt_s":1.0/fps},"environment":{"environment_id":bg_id,"scene_id":f"scene_{si:06d}"},
          "camera":{"camera_id":"front_camera","width_px":W,"height_px":H,"model":"pinhole","intrinsics":{"fx":fx,"fy":fy,"cx":cx,"cy":cy,"K":K.tolist()},"distortion":{"model":"none","coefficients_order":["k1","k2","p1","p2","k3"],"coefficients":[0,0,0,0,0]},"T_body_camera":np.eye(4).tolist()},
          "course":{"gate_tracks":[{"track_id":g["track_id"],"gate_type_id":"standard_gate","route_order_index":g["route_order_index"],"T_world_gate":g["T_world_gate"].tolist()} for g in gates]},
          "domain_randomization":{"enabled":True,"noise_tier":noise["tier"],"sequence_fixed_parameters":{k:(float(v) if isinstance(v,(int,float,np.floating)) else v) for k,v in noise.items()},"background_id":bg_id}
        }
        json_dump(seqdir/"sequence.json",seq_meta)
        records=[]
        for fi,(pos,vel,T_wc) in enumerate(traj):
            frng=np.random.default_rng(base_seed + si*100003 + fi*97 + 17)
            b=load_background_frame(bg,W,H,fi,frames)
            if b is None:
                b=procedural_background(frng,W,H,fi,frames)
            clean,mask,gate_records=render_frame(b,skin,gates,T_wc,K,W,H,cfg["render"]["include_side_faces"],cfg["render"]["side_face_brightness"])
            noisy=apply_photometric(clean,noise,fi,frames,frng)
            rgb_rel=f"rgb/frame_{fi:06d}.jpg"; mask_rel=f"instance_masks/frame_{fi:06d}.png"
            save_rgb(seqdir/rgb_rel,noisy,cfg["image"]["jpeg_quality"]); save_mask(seqdir/mask_rel,mask)

            ahead=[g for g in gates if g["position_world"][0] > pos[0]+0.15]
            current=min(ahead,key=lambda g:g["position_world"][0]) if ahead else None
            nextg=None
            if current is not None:
                later=[g for g in ahead if g["route_order_index"]>current["route_order_index"]]
                nextg=min(later,key=lambda g:g["route_order_index"]) if later else None
            for gr in gate_records:
                gr["is_current_target"] = bool(current and gr["track_id"]==current["track_id"])
            R_wc=T_wc[:3,:3]
            q=Rotation.from_matrix(R_wc).as_quat()
            speed=float(np.linalg.norm(vel))
            record={
              "schema_version":cfg["schema_version"],"sequence_id":seq_id,"frame_index":fi,"frame_id":f"{seq_id}_frame_{fi:06d}","timestamp_ns":int(round(fi/fps*1e9)),"sim_time_s":fi/fps,
              "files":{"rgb":rgb_rel,"instance_mask":mask_rel,"depth_mm":None,"optical_flow":None,"surface_normals":None},"image":{"width_px":W,"height_px":H},
              "camera":{"intrinsics":{"fx":fx,"fy":fy,"cx":cx,"cy":cy,"K":K.tolist()},"T_world_camera":T_wc.tolist(),"position_world_m":[float(v) for v in pos],"quaternion_world_xyzw":[float(v) for v in q]},
              "drone":{"T_world_body":T_wc.tolist(),"position_world_m":[float(v) for v in pos],"quaternion_world_xyzw":[float(v) for v in q],"linear_velocity_world_mps":[float(v) for v in vel],"linear_velocity_body_mps":None,"angular_velocity_body_radps":None,"linear_acceleration_body_mps2":None},
              "route":{"current_target_track_id":current["track_id"] if current else None,"next_target_track_id":nextg["track_id"] if nextg else None,"current_target_route_index":current["route_order_index"] if current else None},
              "gates":gate_records,
              "frame_conditions":{"gate_count_visible":int(sum(1 for g in gate_records if g["visibility"]["has_visible_pixels"])),"gate_count_annotated":len(gate_records),"drone_speed_mps":speed,"render_randomization":{"tier":noise["tier"],"brightness_factor":noise["brightness"],"contrast_factor":noise["contrast"],"gamma":noise["gamma"],"motion_blur_strength":noise["motion_blur_px"],"noise_strength":noise["gaussian_noise_std"],"speckle_std":noise["speckle_std"],"extra":{}}}
            }
            records.append(record)
        write_jsonl(seqdir/"frames.jsonl",records)
    write_checksums(root)
    return root
