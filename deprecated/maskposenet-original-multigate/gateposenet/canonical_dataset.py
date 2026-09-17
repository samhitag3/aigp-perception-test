"""Canonical-data adapter for the *unchanged* legacy GatePoseNetSingle.

The source dataset follows the UAV Gate Perception Dataset Contract v1.0:

    dataset.json
    gate_geometry.json
    splits/{train,validation,test}_sequences.txt
    sequences/<sequence_id>/sequence.json
    sequences/<sequence_id>/frames.jsonl
    sequences/<sequence_id>/rgb/*
    sequences/<sequence_id>/instance_masks/*

This module intentionally converts that rich, multi-gate source-of-truth into
exactly the tensors expected by the original GatePoseNetSingle training code:

* current target gate only for pose/corner regression;
* outer four keypoints only (TL,TR,BR,BL);
* one UNION auxiliary mask derived from the integer instance-ID PNG;
* legacy normalized coordinate convention x/W, y/H;
* [v_cam(3), omega_cam(3), dt] ego input.

The model architecture, losses, and heads are not changed by this adapter.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


OUTER_KEYS = ("outer_tl", "outer_tr", "outer_br", "outer_bl")


@dataclass
class _CanonicalFrame:
    sequence_id: str
    frame_index: int
    timestamp_ns: Optional[int]
    rgb_path: Path
    instance_mask_path: Optional[Path]
    corners_px: Optional[np.ndarray]       # (4,2), legacy outer corners
    position_cam_m: np.ndarray             # (3,)
    R_cam_gate: np.ndarray                 # (3,3)
    depth_camera_z_m: float
    visible: bool
    visible_frac: float
    ego_cam: np.ndarray                    # (6,) [v_cam, omega_cam]
    dt_s: float
    K: np.ndarray                          # (3,3), native image coordinates
    native_wh: tuple[int, int]


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL {path}:{line_no}: {exc}") from exc
    return out


def _as_T(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (4, 4) or not np.isfinite(a).all():
        return None
    return a


def _as_K(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (3, 3) or not np.isfinite(a).all():
        return None
    return a


def _camera_K(frame: dict, seq: dict, width: int, height: int,
              nominal_focal: float) -> np.ndarray:
    fk = ((frame.get("camera") or {}).get("intrinsics") or {}).get("K")
    sk = ((seq.get("camera") or {}).get("intrinsics") or {}).get("K")
    K = _as_K(fk)
    if K is None:
        K = _as_K(sk)
    if K is None:
        # Contract fallback; for the AIGP camera this becomes exactly
        # fx=fy=320, cx=320, cy=180 at 640x360.
        K = np.array([[nominal_focal, 0.0, width / 2.0],
                      [0.0, nominal_focal, height / 2.0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
    return K.astype(np.float32)


def _target_gate(frame: dict) -> Optional[dict]:
    gates = frame.get("gates") or []
    if not gates:
        return None

    route = frame.get("route") or {}
    target_id = route.get("current_target_track_id")
    if target_id is not None:
        for gate in gates:
            if gate.get("track_id") == target_id:
                return gate

    for gate in gates:
        if bool(gate.get("is_current_target", False)):
            return gate

    route_idx = route.get("current_target_route_index")
    if route_idx is not None:
        for gate in gates:
            if gate.get("route_order_index") == route_idx:
                return gate

    # Last-resort compatibility for datasets that have not yet populated
    # route metadata. This is deterministic and does not reorder by visibility.
    return min(gates, key=lambda g: (
        g.get("route_order_index") is None,
        g.get("route_order_index", 1 << 30),
        str(g.get("track_id", "")),
    ))


def _sequence_gate_track(seq: dict, track_id: Optional[str]) -> Optional[dict]:
    if track_id is None:
        return None
    for g in ((seq.get("course") or {}).get("gate_tracks") or []):
        if g.get("track_id") == track_id:
            return g
    return None


def _gate_geometry_points(gate_geometry: dict, gate_type_id: str) -> Optional[np.ndarray]:
    gt = (gate_geometry.get("gate_types") or {}).get(gate_type_id)
    if not gt:
        return None
    kp = gt.get("keypoints_gate_frame_m") or {}
    vals = []
    for key in OUTER_KEYS:
        if kp.get(key) is None:
            return None
        vals.append(kp[key])
    a = np.asarray(vals, dtype=np.float64)
    return a if a.shape == (4, 3) else None


def _project(points_gate: np.ndarray, T_camera_gate: np.ndarray,
             K: np.ndarray) -> Optional[np.ndarray]:
    p_h = np.concatenate([points_gate, np.ones((len(points_gate), 1))], axis=1)
    p_cam = (T_camera_gate @ p_h.T).T[:, :3]
    if (p_cam[:, 2] <= 1e-8).any():
        # Keep projections undefined when any requested corner is behind the
        # pinhole. GatePoseNet's corner_ok will then correctly become zero.
        return None
    uvw = (K.astype(np.float64) @ p_cam.T).T
    return uvw[:, :2] / uvw[:, 2:3]


def _corners_from_gate(gate: dict) -> Optional[np.ndarray]:
    kp = gate.get("keypoints_2d") or {}
    vals = []
    for key in OUTER_KEYS:
        rec = kp.get(key) or {}
        xy = rec.get("projected_px")
        if xy is None:
            return None
        vals.append(xy)
    a = np.asarray(vals, dtype=np.float64)
    return a if a.shape == (4, 2) and np.isfinite(a).all() else None


def _pose_from_gate(gate: dict, frame: dict, seq: dict) -> tuple[Optional[np.ndarray], Optional[str], str]:
    pose = gate.get("pose") or {}
    Tcg = _as_T(pose.get("T_camera_gate"))
    gate_type = str(gate.get("gate_type_id") or "standard_gate")
    track_id = gate.get("track_id")
    if Tcg is not None:
        return Tcg, track_id, gate_type

    # Derive camera-relative pose from world transforms if the frame does not
    # redundantly store T_camera_gate.
    Twc = _as_T((frame.get("camera") or {}).get("T_world_camera"))
    Twg = _as_T(pose.get("T_world_gate"))
    if Twg is None:
        sg = _sequence_gate_track(seq, track_id)
        if sg is not None:
            Twg = _as_T(sg.get("T_world_gate"))
            gate_type = str(sg.get("gate_type_id") or gate_type)
    if Twc is not None and Twg is not None:
        return np.linalg.inv(Twc) @ Twg, track_id, gate_type
    return None, track_id, gate_type


def _visibility_from_gate(gate: dict) -> tuple[bool, float]:
    v = gate.get("visibility") or {}
    has_pixels = v.get("has_visible_pixels")
    visible_px = v.get("visible_pixel_area")

    # Legacy GatePoseNet's visible_frac means fraction of the *full projected
    # silhouette* that is actually visible/in-frame. The canonical contract
    # separates occlusion and truncation, so reconstruct the legacy quantity.
    full_px = v.get("amodal_area_full_px")
    if visible_px is not None and full_px is not None and float(full_px) > 0:
        frac = float(visible_px) / float(full_px)
    else:
        inframe_visible = v.get("visible_fraction")
        trunc = v.get("truncation_fraction")
        if inframe_visible is not None:
            frac = float(inframe_visible) * (1.0 - float(trunc or 0.0))
        else:
            frac = 1.0 if bool(has_pixels) else 0.0
    frac = float(np.clip(float(frac), 0.0, 1.0))
    if has_pixels is None:
        has_pixels = (float(visible_px or 0) > 0) or frac > 0.0
    return bool(has_pixels), frac


def _rotation_vector_from_matrix(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    return rvec.reshape(3)


def _ego_from_states(frame: dict) -> Optional[np.ndarray]:
    """Convert canonical drone velocities to the legacy camera-frame ego6.

    The contract provides world-frame linear velocity and body-frame angular
    velocity. The current frame transforms are enough to rotate both into the
    camera optical frame. Lever-arm translational velocity from camera/body
    offset is not added; if exact camera ego is supplied as `ego_motion_cam`,
    that explicit field takes precedence.
    """
    explicit = frame.get("ego_motion_cam")
    if explicit is not None:
        a = np.asarray(explicit, dtype=np.float64)
        if a.shape == (6,) and np.isfinite(a).all():
            return a.astype(np.float32)

    drone = frame.get("drone") or {}
    v_world = drone.get("linear_velocity_world_mps")
    w_body = drone.get("angular_velocity_body_radps")
    Twc = _as_T((frame.get("camera") or {}).get("T_world_camera"))
    Twb = _as_T(drone.get("T_world_body"))
    if Twc is None:
        return None

    Rwc = Twc[:3, :3]
    out = np.zeros(6, dtype=np.float64)
    have = False
    if v_world is not None:
        v = np.asarray(v_world, dtype=np.float64)
        if v.shape == (3,) and np.isfinite(v).all():
            out[:3] = Rwc.T @ v
            have = True
    if w_body is not None and Twb is not None:
        w = np.asarray(w_body, dtype=np.float64)
        if w.shape == (3,) and np.isfinite(w).all():
            Rwb = Twb[:3, :3]
            Rcb = Rwc.T @ Rwb
            out[3:] = Rcb @ w
            have = True
    return out.astype(np.float32) if have else None


def _finite_difference_ego(frames: list[dict], i: int, dt: float) -> np.ndarray:
    if i <= 0 or dt <= 0:
        return np.zeros(6, dtype=np.float32)
    prev = _as_T((frames[i - 1].get("camera") or {}).get("T_world_camera"))
    cur = _as_T((frames[i].get("camera") or {}).get("T_world_camera"))
    if prev is None or cur is None:
        return np.zeros(6, dtype=np.float32)

    Rwc = cur[:3, :3]
    v_world = (cur[:3, 3] - prev[:3, 3]) / dt
    v_cam = Rwc.T @ v_world

    # Relative camera orientation over the step. Rodrigues returns the compact
    # axis-angle vector. Convert from previous-camera coordinates to current.
    R_prev_to_cur_axes = prev[:3, :3].T @ cur[:3, :3]
    w_prev = _rotation_vector_from_matrix(R_prev_to_cur_axes) / dt
    w_cam = (cur[:3, :3].T @ prev[:3, :3]) @ w_prev
    return np.concatenate([v_cam, w_cam]).astype(np.float32)


def _timestamp_seconds(frame: dict) -> Optional[float]:
    if frame.get("timestamp_ns") is not None:
        return float(frame["timestamp_ns"]) * 1e-9
    if frame.get("sim_time_s") is not None:
        return float(frame["sim_time_s"])
    if frame.get("timestamp_s") is not None:
        return float(frame["timestamp_s"])
    return None


class CanonicalGateSequenceDataset(Dataset):
    """Rolling-window adapter from the canonical dataset to GatePoseNetSingle."""

    def __init__(
        self,
        dataset_root: str | Path,
        manifest: str | Path,
        window: int = 8,
        stride: int = 4,
        size: tuple[int, int] = (192, 320),
        pose_sup_max_m: float = 20.0,
        nominal_focal: float = 320.0,
        load_masks: bool = True,
        ego_dropout: float = 0.0,
    ):
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.size = (int(size[0]), int(size[1]))
        self.window = int(window)
        self.stride = max(1, int(stride))
        self.pose_sup_max_m = float(pose_sup_max_m)
        self.nominal_focal = float(nominal_focal)
        self.load_masks = bool(load_masks)
        self.ego_dropout = float(ego_dropout)

        manifest_path = Path(manifest)
        if not manifest_path.is_absolute():
            manifest_path = self.dataset_root / manifest_path
        if not manifest_path.exists():
            raise FileNotFoundError(f"sequence manifest not found: {manifest_path}")
        sequence_ids = [x.strip() for x in manifest_path.read_text().splitlines()
                        if x.strip() and not x.lstrip().startswith("#")]
        if not sequence_ids:
            raise RuntimeError(f"empty sequence manifest: {manifest_path}")

        gg_path = self.dataset_root / "gate_geometry.json"
        self.gate_geometry = _read_json(gg_path) if gg_path.exists() else {}

        self._frames: list[_CanonicalFrame] = []
        self._windows: list[list[int]] = []
        self._window_meta: list[dict[str, Any]] = []

        for sequence_id in sequence_ids:
            self._add_sequence(sequence_id)

        if not self._windows:
            raise RuntimeError(
                f"no usable GatePoseNet windows in {manifest_path}; "
                "check target-gate pose annotations and RGB paths")

    def _add_sequence(self, sequence_id: str) -> None:
        seq_dir = self.dataset_root / "sequences" / sequence_id
        seq_path = seq_dir / "sequence.json"
        frames_path = seq_dir / "frames.jsonl"
        if not seq_path.exists() or not frames_path.exists():
            raise FileNotFoundError(
                f"canonical sequence {sequence_id!r} missing sequence.json or frames.jsonl")
        seq = _read_json(seq_path)
        raw_frames = _read_jsonl(frames_path)
        raw_frames.sort(key=lambda f: int(f.get("frame_index", 0)))

        seq_camera = seq.get("camera") or {}
        seq_w = int(seq_camera.get("width_px") or 640)
        seq_h = int(seq_camera.get("height_px") or 360)
        nominal_fps = float((seq.get("timing") or {}).get("nominal_fps") or 30.0)
        default_dt = 1.0 / max(nominal_fps, 1e-6)

        local_idxs: list[int] = []
        for i, frame in enumerate(raw_frames):
            files = frame.get("files") or {}
            rgb_rel = files.get("rgb")
            if not rgb_rel:
                continue
            rgb_path = seq_dir / rgb_rel

            image_meta = frame.get("image") or {}
            W0 = int(image_meta.get("width_px") or seq_w)
            H0 = int(image_meta.get("height_px") or seq_h)
            K = _camera_K(frame, seq, W0, H0, self.nominal_focal)

            gate = _target_gate(frame)
            route = frame.get("route") or {}
            route_target = route.get("current_target_track_id")

            if gate is None and route_target is not None:
                sg = _sequence_gate_track(seq, route_target)
                if sg is not None:
                    # Synthesize the minimum observation from known world GT.
                    gate = {
                        "track_id": route_target,
                        "gate_type_id": sg.get("gate_type_id", "standard_gate"),
                        "route_order_index": sg.get("route_order_index"),
                        "visibility": {
                            "has_visible_pixels": False,
                            "visible_fraction": 0.0,
                            "visible_pixel_area": 0,
                        },
                        "pose": {"T_world_gate": sg.get("T_world_gate")},
                    }
            if gate is None:
                continue

            Tcg, track_id, gate_type = _pose_from_gate(gate, frame, seq)
            if Tcg is None:
                continue
            R = Tcg[:3, :3].astype(np.float32)
            pos = Tcg[:3, 3].astype(np.float32)
            depth = float(pos[2])

            corners = _corners_from_gate(gate)
            if corners is None:
                pts_gate = _gate_geometry_points(self.gate_geometry, gate_type)
                if pts_gate is not None:
                    corners = _project(pts_gate, Tcg, K)

            visible, visible_frac = _visibility_from_gate(gate)

            if i > 0:
                t_now = _timestamp_seconds(raw_frames[i])
                t_prev = _timestamp_seconds(raw_frames[i - 1])
                dt = (t_now - t_prev) if t_now is not None and t_prev is not None else default_dt
                if not np.isfinite(dt) or dt <= 0:
                    dt = default_dt
            else:
                dt = default_dt

            ego6 = _ego_from_states(frame)
            if ego6 is None:
                ego6 = _finite_difference_ego(raw_frames, i, dt)

            mask_rel = files.get("instance_mask")
            mask_path = (seq_dir / mask_rel) if mask_rel else None

            rec = _CanonicalFrame(
                sequence_id=sequence_id,
                frame_index=int(frame.get("frame_index", i)),
                timestamp_ns=(int(frame["timestamp_ns"])
                              if frame.get("timestamp_ns") is not None else None),
                rgb_path=rgb_path,
                instance_mask_path=mask_path,
                corners_px=(corners.astype(np.float64) if corners is not None else None),
                position_cam_m=pos,
                R_cam_gate=R,
                depth_camera_z_m=depth,
                visible=visible,
                visible_frac=visible_frac,
                ego_cam=np.asarray(ego6, dtype=np.float32),
                dt_s=float(dt),
                K=K,
                native_wh=(W0, H0),
            )
            local_idxs.append(len(self._frames))
            self._frames.append(rec)

        if not local_idxs:
            return
        if len(local_idxs) < self.window:
            self._windows.append(local_idxs)
            self._window_meta.append({"sequence_id": sequence_id,
                                      "frame_indices": [self._frames[j].frame_index for j in local_idxs]})
            return

        starts = list(range(0, len(local_idxs) - self.window + 1, self.stride))
        if starts[-1] != len(local_idxs) - self.window:
            starts.append(len(local_idxs) - self.window)
        for s in starts:
            win = local_idxs[s:s + self.window]
            self._windows.append(win)
            self._window_meta.append({"sequence_id": sequence_id,
                                      "frame_indices": [self._frames[j].frame_index for j in win]})

    def __len__(self) -> int:
        return len(self._windows)

    def window_meta(self, index: int) -> dict[str, Any]:
        return dict(self._window_meta[index])

    def _load_frame(self, rec: _CanonicalFrame) -> dict[str, torch.Tensor]:
        h, w = self.size
        img = cv2.imread(str(rec.rgb_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(rec.rgb_path)
        H_img, W_img = img.shape[:2]
        W0, H0 = rec.native_wh
        if (W_img, H_img) != (W0, H0):
            # Use actual bytes as authoritative for 2-D normalization, while K
            # is still expected to have been generated for this image size.
            W0, H0 = W_img, H_img

        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        # Legacy GatePoseNet aux head is UNION segmentation. The canonical PNG
        # stays instance-ID encoded; the union is derived losslessly here.
        seg_t = torch.zeros(1, h // 4, w // 4)
        if self.load_masks and rec.instance_mask_path is not None:
            m = cv2.imread(str(rec.instance_mask_path), cv2.IMREAD_UNCHANGED)
            if m is None:
                raise FileNotFoundError(rec.instance_mask_path)
            if m.ndim == 3:
                m = m[..., 0]
            union = (m > 0).astype(np.float32)
            union = cv2.resize(union, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
            seg_t = torch.from_numpy((union > 0.5).astype(np.float32))[None]

        in_front = rec.depth_camera_z_m > 0.05
        corners_uv = np.zeros((4, 2), dtype=np.float32)
        corner_ok = np.zeros(4, dtype=np.float32)
        corner_inside = np.zeros(4, dtype=np.float32)
        if rec.corners_px is not None and in_front:
            # IMPORTANT: the original GatePoseNet uses x/W and y/H (not the
            # canonical JSON's optional x/(W-1), y/(H-1) convenience field).
            # Converting from projected_px here preserves the legacy model's
            # exact target convention without changing canonical source data.
            cn = rec.corners_px / np.array([W0, H0], dtype=np.float64)
            corners_uv = cn.astype(np.float32)
            sane = ((cn[:, 0] > -1.0) & (cn[:, 0] < 2.0)
                    & (cn[:, 1] > -1.0) & (cn[:, 1] < 2.0))
            corner_ok = sane.astype(np.float32)
            inside = ((rec.corners_px[:, 0] >= 0) & (rec.corners_px[:, 0] < W0)
                      & (rec.corners_px[:, 1] >= 0) & (rec.corners_px[:, 1] < H0))
            corner_inside = inside.astype(np.float32)

        center_uv = np.zeros(2, dtype=np.float32)
        center_ok = 0.0
        if in_front:
            x, y, z = [float(v) for v in rec.position_cam_m]
            center_px = np.array([
                rec.K[0, 0] * x / z + rec.K[0, 2],
                rec.K[1, 1] * y / z + rec.K[1, 2],
            ], dtype=np.float64)
            cn = center_px / np.array([W0, H0], dtype=np.float64)
            if -1.0 < cn[0] < 2.0 and -1.0 < cn[1] < 2.0:
                center_uv = cn.astype(np.float32)
                center_ok = 1.0

        dist = float(np.linalg.norm(rec.position_cam_m))
        pose_ok = 1.0 if dist <= self.pose_sup_max_m else 0.0
        log_depth = (float(math.log(max(rec.depth_camera_z_m, 0.05)))
                     if rec.depth_camera_z_m > 0.05 else 0.0)
        depth_ok = 1.0 if rec.depth_camera_z_m > 0.05 and pose_ok > 0 else 0.0

        k_scale = np.array([rec.K[0, 0], rec.K[1, 1]], dtype=np.float32) \
            / self.nominal_focal
        ego = np.concatenate([rec.ego_cam, [np.float32(rec.dt_s)]]).astype(np.float32)

        return dict(
            image=img_t,
            seg=seg_t,
            corners_uv=torch.from_numpy(corners_uv),
            corner_ok=torch.from_numpy(corner_ok),
            corner_inside=torch.from_numpy(corner_inside),
            center_uv=torch.from_numpy(center_uv),
            center_ok=torch.tensor(center_ok, dtype=torch.float32),
            position=torch.from_numpy(rec.position_cam_m.astype(np.float32)),
            R=torch.from_numpy(rec.R_cam_gate.astype(np.float32)),
            log_depth=torch.tensor(log_depth, dtype=torch.float32),
            depth_ok=torch.tensor(depth_ok, dtype=torch.float32),
            pose_ok=torch.tensor(pose_ok, dtype=torch.float32),
            visible=torch.tensor(1.0 if rec.visible else 0.0),
            visible_frac=torch.tensor(rec.visible_frac, dtype=torch.float32),
            ego=torch.from_numpy(ego),
            k_scale=torch.from_numpy(k_scale),
            K=torch.from_numpy(rec.K.astype(np.float32)),
            img_wh=torch.tensor([float(W0), float(H0)], dtype=torch.float32),
        )

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        idxs = self._windows[i]
        n_pad = max(0, self.window - len(idxs))
        padded = [idxs[0]] * n_pad + list(idxs)
        frames = [self._load_frame(self._frames[j]) for j in padded]
        out = {k: torch.stack([f[k] for f in frames], dim=0)
               for k in frames[0]}
        out["frame_ok"] = torch.cat([
            torch.zeros(n_pad, dtype=torch.float32),
            torch.ones(len(padded) - n_pad, dtype=torch.float32),
        ])
        if n_pad:
            out["ego"][:n_pad, :6] = 0.0
        ego_available = 1.0
        if self.ego_dropout > 0 and torch.rand(()) < self.ego_dropout:
            out["ego"][:, :6] = 0.0
            ego_available = 0.0
        out["ego_ok"] = torch.full((self.window,), ego_available,
                                    dtype=torch.float32)
        return out
