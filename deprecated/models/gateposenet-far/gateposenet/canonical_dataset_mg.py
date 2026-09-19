"""Canonical multi-gate adapter for the unchanged GatePoseNet-MG architecture.

Reads the UAV Gate Perception Dataset Contract v1.x directly:
  dataset.json
  gate_geometry.json
  splits/*.txt
  sequences/<id>/sequence.json
  sequences/<id>/frames.jsonl
  sequences/<id>/rgb/*
  sequences/<id>/instance_masks/*

The canonical integer instance PNG remains the source of truth.  For each gate,
its binary supervision mask is derived as ``instance_mask == gate.mask_id``.
All gates with valid camera-relative pose are supervised up to ``max_gates``;
the current route target is always prioritized when truncation is necessary.

GatePoseNet-MG itself is NOT modified by this adapter.  It still predicts four
outer corners per gate, one mask per query, 6-DoF pose, presence and target score.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .canonical_dataset import (
    OUTER_KEYS,
    _as_T,
    _camera_K,
    _corners_from_gate,
    _ego_from_states,
    _finite_difference_ego,
    _gate_geometry_points,
    _pose_from_gate,
    _project,
    _read_json,
    _read_jsonl,
    _sequence_gate_track,
    _timestamp_seconds,
    _visibility_from_gate,
)


@dataclass
class _MGGate:
    track_id: Optional[str]
    mask_id: Optional[int]
    route_order_index: Optional[int]
    gate_type_id: str
    corners_px: Optional[np.ndarray]
    position_cam_m: np.ndarray
    R_cam_gate: np.ndarray
    depth_camera_z_m: float
    visible: bool
    visible_frac: float
    is_target: bool


@dataclass
class _MGFrame:
    sequence_id: str
    frame_index: int
    timestamp_ns: Optional[int]
    rgb_path: Path
    instance_mask_path: Optional[Path]
    gates: list[_MGGate]
    ego_cam: np.ndarray
    dt_s: float
    K: np.ndarray
    native_wh: tuple[int, int]


def _route_target_id(frame: dict) -> Optional[str]:
    route = frame.get("route") or {}
    tid = route.get("current_target_track_id")
    if tid is not None:
        return str(tid)
    for g in frame.get("gates") or []:
        if bool(g.get("is_current_target", False)):
            return str(g.get("track_id")) if g.get("track_id") is not None else None
    ridx = route.get("current_target_route_index")
    if ridx is not None:
        for g in frame.get("gates") or []:
            if g.get("route_order_index") == ridx:
                return str(g.get("track_id")) if g.get("track_id") is not None else None
    return None


def _synth_target_if_missing(frame: dict, seq: dict, target_id: Optional[str]) -> list[dict]:
    gates = [dict(g) for g in (frame.get("gates") or [])]
    if target_id is None or any(str(g.get("track_id")) == target_id for g in gates):
        return gates
    sg = _sequence_gate_track(seq, target_id)
    if sg is None:
        return gates
    gates.append({
        "track_id": target_id,
        "gate_type_id": sg.get("gate_type_id", "standard_gate"),
        "route_order_index": sg.get("route_order_index"),
        "mask_id": None,
        "visibility": {
            "has_visible_pixels": False,
            "visible_pixel_area": 0,
            "visible_fraction": 0.0,
            "truncation_fraction": 0.0,
        },
        "pose": {"T_world_gate": sg.get("T_world_gate")},
        "is_current_target": True,
    })
    return gates


class CanonicalGateSequenceDatasetMG(Dataset):
    """Rolling-window canonical adapter that supervises all gate instances."""

    def __init__(
        self,
        dataset_root: str | Path,
        manifest: str | Path,
        window: int = 8,
        stride: int = 4,
        size: tuple[int, int] = (192, 320),
        pose_sup_max_m: float = 25.0,
        nominal_focal: float = 320.0,
        ego_dropout: float = 0.0,
        max_gates: int = 8,
        mask_stride: int = 4,
        preserve_small_masks: bool = False,
    ):
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.size = (int(size[0]), int(size[1]))
        self.window = int(window)
        self.stride = max(1, int(stride))
        self.pose_sup_max_m = float(pose_sup_max_m)
        self.nominal_focal = float(nominal_focal)
        self.ego_dropout = float(ego_dropout)
        self.max_gates = int(max_gates)
        self.mask_stride = int(mask_stride)
        self.preserve_small_masks = bool(preserve_small_masks)
        if self.max_gates < 1:
            raise ValueError("max_gates must be >= 1")
        if self.mask_stride not in (2, 4):
            raise ValueError("mask_stride must be 2 or 4")

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

        self._frames: list[_MGFrame] = []
        self._windows: list[list[int]] = []
        self._window_meta: list[dict[str, Any]] = []
        self.dropped_gates_due_to_limit = 0

        for sequence_id in sequence_ids:
            self._add_sequence(sequence_id)
        if not self._windows:
            raise RuntimeError(f"no usable MG windows in {manifest_path}")

    def _gate_record(self, g: dict, frame: dict, seq: dict, K: np.ndarray,
                     target_id: Optional[str]) -> Optional[_MGGate]:
        Tcg, track_id, gate_type = _pose_from_gate(g, frame, seq)
        if Tcg is None:
            return None
        pos = Tcg[:3, 3].astype(np.float32)
        if not np.isfinite(pos).all():
            return None
        dist = float(np.linalg.norm(pos))
        if dist > self.pose_sup_max_m:
            return None
        R = Tcg[:3, :3].astype(np.float32)
        depth = float(pos[2])

        corners = _corners_from_gate(g)
        if corners is None:
            pts = _gate_geometry_points(self.gate_geometry, gate_type)
            if pts is not None:
                corners = _project(pts, Tcg, K)
        visible, visible_frac = _visibility_from_gate(g)
        mid = g.get("mask_id")
        try:
            mid = int(mid) if mid is not None else None
        except (TypeError, ValueError):
            mid = None
        tid = str(track_id) if track_id is not None else None
        return _MGGate(
            track_id=tid,
            mask_id=mid,
            route_order_index=(int(g["route_order_index"])
                               if g.get("route_order_index") is not None else None),
            gate_type_id=str(gate_type),
            corners_px=(corners.astype(np.float64) if corners is not None else None),
            position_cam_m=pos,
            R_cam_gate=R,
            depth_camera_z_m=depth,
            visible=visible,
            visible_frac=visible_frac,
            is_target=(target_id is not None and tid == target_id)
                      or bool(g.get("is_current_target", False)),
        )

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

            target_id = _route_target_id(frame)
            gate_dicts = _synth_target_if_missing(frame, seq, target_id)
            gates = []
            for gd in gate_dicts:
                rec = self._gate_record(gd, frame, seq, K, target_id)
                if rec is not None:
                    gates.append(rec)

            # Target first, then gates with visible pixels, then route order,
            # then range. This keeps truncation deterministic and flight-aware.
            gates.sort(key=lambda g: (
                not g.is_target,
                not g.visible,
                g.route_order_index is None,
                g.route_order_index if g.route_order_index is not None else 1 << 30,
                float(np.linalg.norm(g.position_cam_m)),
                g.track_id or "",
            ))
            if len(gates) > self.max_gates:
                self.dropped_gates_due_to_limit += len(gates) - self.max_gates
                gates = gates[:self.max_gates]
            if gates and not any(g.is_target for g in gates):
                # Legacy-compatible deterministic fallback when route metadata is absent.
                gates[0].is_target = True

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
            rec = _MGFrame(
                sequence_id=sequence_id,
                frame_index=int(frame.get("frame_index", i)),
                timestamp_ns=(int(frame["timestamp_ns"])
                              if frame.get("timestamp_ns") is not None else None),
                rgb_path=rgb_path,
                instance_mask_path=mask_path,
                gates=gates,
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

    def _load_frame(self, rec: _MGFrame) -> dict[str, torch.Tensor]:
        h, w = self.size
        img = cv2.imread(str(rec.rgb_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(rec.rgb_path)
        H_img, W_img = img.shape[:2]
        W0, H0 = rec.native_wh
        if (W_img, H_img) != (W0, H0):
            W0, H0 = W_img, H_img
        img_r = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        image = torch.from_numpy(cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
                                 .transpose(2, 0, 1)).float() / 255.0

        mask_ids = None
        if rec.instance_mask_path is not None and rec.instance_mask_path.exists():
            mask_ids = cv2.imread(str(rec.instance_mask_path), cv2.IMREAD_UNCHANGED)
            if mask_ids is None:
                raise FileNotFoundError(rec.instance_mask_path)
            if mask_ids.ndim == 3:
                mask_ids = mask_ids[..., 0]

        N = self.max_gates
        g = dict(
            g_valid=torch.zeros(N),
            g_mask=torch.zeros(N, h // self.mask_stride, w // self.mask_stride),
            g_corners=torch.zeros(N, 4, 2),
            g_corner_ok=torch.zeros(N, 4),
            g_inside=torch.zeros(N, 4),
            g_center=torch.zeros(N, 2),
            g_center_ok=torch.zeros(N),
            g_position=torch.zeros(N, 3),
            g_R=torch.eye(3).repeat(N, 1, 1),
            g_log_depth=torch.zeros(N),
            g_depth_ok=torch.zeros(N),
            g_visible=torch.zeros(N),
            g_visible_frac=torch.zeros(N),
            g_target=torch.zeros(N),
        )

        for i, gate in enumerate(rec.gates[:N]):
            g["g_valid"][i] = 1.0
            if mask_ids is not None and gate.mask_id is not None and gate.mask_id > 0:
                bm = (mask_ids == gate.mask_id).astype(np.uint8)
                target_wh = (w // self.mask_stride, h // self.mask_stride)
                if self.preserve_small_masks:
                    # Area resize followed by >0 occupancy preserves thin gate
                    # bars that nearest-neighbor downsampling can erase.
                    occ = cv2.resize(bm.astype(np.float32), target_wh,
                                     interpolation=cv2.INTER_AREA)
                    bm = (occ > 0.0).astype(np.float32)
                else:
                    bm = cv2.resize(bm, target_wh, interpolation=cv2.INTER_NEAREST).astype(np.float32)
                g["g_mask"][i] = torch.from_numpy(bm)

            in_front = gate.depth_camera_z_m > 0.05
            if gate.corners_px is not None and in_front:
                cn = gate.corners_px / np.array([W0, H0], dtype=np.float64)
                g["g_corners"][i] = torch.from_numpy(cn.astype(np.float32))
                sane = ((cn[:, 0] > -1.0) & (cn[:, 0] < 2.0)
                        & (cn[:, 1] > -1.0) & (cn[:, 1] < 2.0))
                g["g_corner_ok"][i] = torch.from_numpy(sane.astype(np.float32))
                inside = ((gate.corners_px[:, 0] >= 0) & (gate.corners_px[:, 0] < W0)
                          & (gate.corners_px[:, 1] >= 0) & (gate.corners_px[:, 1] < H0))
                g["g_inside"][i] = torch.from_numpy(inside.astype(np.float32))

            if in_front:
                x, y, z = [float(v) for v in gate.position_cam_m]
                center_px = np.array([
                    rec.K[0, 0] * x / z + rec.K[0, 2],
                    rec.K[1, 1] * y / z + rec.K[1, 2],
                ], dtype=np.float64)
                cn = center_px / np.array([W0, H0], dtype=np.float64)
                if -1.0 < cn[0] < 2.0 and -1.0 < cn[1] < 2.0:
                    g["g_center"][i] = torch.from_numpy(cn.astype(np.float32))
                    g["g_center_ok"][i] = 1.0

            g["g_position"][i] = torch.from_numpy(gate.position_cam_m.astype(np.float32))
            g["g_R"][i] = torch.from_numpy(gate.R_cam_gate.astype(np.float32))
            if in_front:
                g["g_log_depth"][i] = float(math.log(max(gate.depth_camera_z_m, 0.05)))
                g["g_depth_ok"][i] = 1.0
            g["g_visible"][i] = 1.0 if gate.visible else 0.0
            g["g_visible_frac"][i] = float(gate.visible_frac)
            g["g_target"][i] = 1.0 if gate.is_target else 0.0

        if g["g_target"].sum() == 0 and g["g_valid"].sum() > 0:
            first = int(torch.nonzero(g["g_valid"] > 0.5, as_tuple=False)[0])
            g["g_target"][first] = 1.0

        ego = np.concatenate([rec.ego_cam, [np.float32(rec.dt_s)]]).astype(np.float32)
        return dict(image=image, ego=torch.from_numpy(ego), **g)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        idxs = self._windows[index]
        n_pad = max(0, self.window - len(idxs))
        padded = [idxs[0]] * n_pad + list(idxs)
        frames = [self._load_frame(self._frames[j]) for j in padded]
        out = {k: torch.stack([f[k] for f in frames], dim=0) for k in frames[0]}
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
        out["ego_ok"] = torch.full((self.window,), ego_available, dtype=torch.float32)
        return out
