"""Rolling-window sequence dataset over sam3-autolabeler ground_truth.json.

Consumes the trajectory datasets written by `python -m synthetic gen
--trajectories N` (sam3-autolabeler-private): per-object records carrying the
exact analytic GT — corners / center_px / bbox (2-D, pixels), position_cam /
R_cam_gate / distance (3-D, camera optical frame), visible / visible_frac,
seq_id / seq_t, per-sequence K (calibration-shift DR) and exact ego_motion_cam.

Each __getitem__ returns a WINDOW of ``window`` consecutive frames from one
sequence (i.i.d. datasets — seq_id null — degrade gracefully to windows of 1):

    image      (T, 3, H, W) float32 in [0,1]  (resized model input)
    ego        (T, 6)  float32   [v_cam(3), omega_cam(3)] (zeros if absent)
    seg        (T, 1, H/4, W/4) float32       union gate mask (aux supervision)
    corners_uv (T, 4, 2) float32              normalized [0,1] image coords
    corner_ok  (T, 4)  float32                corner supervision mask
    center_uv  (T, 2)  float32                normalized image coords
    center_ok  (T,)   float32                 center supervision mask
    position   (T, 3)  float32                gate center, camera meters
    R          (T, 3, 3) float32              R_cam_gate
    log_depth  (T,)   float32                 log(center_depth_m) (clamped)
    pose_ok    (T,)   float32                 3-D pose supervision mask
    visible    (T,)   float32                 visibility flag GT
    visible_frac (T,) float32                 exact in-frame silhouette frac
    k_scale    (T, 2) float32                 (fx, fy) / nominal (aux target)

2-D targets are normalized by IMAGE SIZE (not K), so the model never assumes a
calibration; the per-sequence K jitter in the data makes it robust to
calibration shift. 3-D pose supervision (`pose_ok`) stays on for frames where
the gate is within ``pose_sup_max_m`` in front of or around the camera — that
INCLUDES invisible fly-through frames, which is what forces the recurrent
state (+ ego-motion) to carry the gate pose through blind moments
(SkyDreamer-style privileged decoding).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class _FrameRec:
    file: str
    corners: Optional[np.ndarray]      # (4,2) px or None
    center_px: Optional[np.ndarray]    # (2,) px or None
    position: np.ndarray               # (3,) m
    R: np.ndarray                      # (3,3)
    depth: float                       # center_depth_m (can be <= 0 behind)
    visible: bool
    visible_frac: float
    ego: Optional[np.ndarray]          # (6,) or None
    K: Optional[np.ndarray]            # (3,3) or None
    seq_id: Optional[int]
    seq_t: Optional[int]


def _load_ground_truth(root: str) -> tuple[list[_FrameRec], dict]:
    gt_path = os.path.join(root, "ground_truth.json")
    with open(gt_path) as f:
        gt = json.load(f)
    recs: list[_FrameRec] = []
    for o in gt.get("objects", []):
        if o.get("position_cam") is None or o.get("R_cam_gate") is None:
            continue
        # Pose supervision targets the gate being flown AT: prefer the
        # per-frame is_target flag (maneuvers switch targets mid-sequence);
        # legacy datasets without it fall back to obj_id 0. Distractor gates
        # still appear in the image and the union seg mask (aux supervision).
        if o.get("is_target") is not None:
            if not o["is_target"]:
                continue
        elif int(o.get("obj_id", 0)) != 0:
            continue
        recs.append(_FrameRec(
            file=o["file"],
            corners=(np.asarray(o["corners"], dtype=np.float64)
                     if o.get("corners") is not None else None),
            center_px=(np.asarray(o["center_px"], dtype=np.float64)
                       if o.get("center_px") is not None else None),
            position=np.asarray(o["position_cam"], dtype=np.float64),
            R=np.asarray(o["R_cam_gate"], dtype=np.float64),
            depth=float(o.get("center_depth_m", 0.0) or 0.0),
            visible=bool(o.get("visible", True)),
            visible_frac=float(o["visible_frac"]) if o.get("visible_frac")
            is not None else (1.0 if o.get("visible", True) else 0.0),
            ego=(np.asarray(o["ego_motion_cam"], dtype=np.float64)
                 if o.get("ego_motion_cam") is not None else None),
            K=(np.asarray(o["K"], dtype=np.float64)
               if o.get("K") is not None else None),
            seq_id=o.get("seq_id"),
            seq_t=o.get("seq_t"),
        ))
    return recs, gt


class GateSequenceDataset(Dataset):
    """Rolling windows of T frames over one or more trajectory datasets."""

    def __init__(
        self,
        roots: str | list[str],
        window: int = 8,
        stride: int = 4,
        size: tuple[int, int] = (192, 320),   # (H, W) model input
        pose_sup_max_m: float = 20.0,
        nominal_focal: float = 320.0,
        load_masks: bool = True,
        ego_dropout: float = 0.0,             # train-time modality dropout
    ):
        if isinstance(roots, str):
            roots = [roots]
        self.size = (int(size[0]), int(size[1]))
        self.window = int(window)
        self.stride = max(1, int(stride))
        self.pose_sup_max_m = float(pose_sup_max_m)
        self.nominal_focal = float(nominal_focal)
        self.load_masks = bool(load_masks)
        self.ego_dropout = float(ego_dropout)

        # frames grouped by (root, seq_id); i.i.d. frames become 1-windows.
        self._frames: list[tuple[str, _FrameRec]] = []
        self._windows: list[list[int]] = []
        self._fps: dict[str, float] = {}
        for root in roots:
            recs, gt_top = _load_ground_truth(root)
            # sequence frame rate -> explicit dt model input (rate-aware
            # temporal state: 90-120 Hz deployment vs 30 Hz sim stream).
            self._fps[root] = float(gt_top.get("fps") or 30.0)
            by_seq: dict = {}
            iid: list[int] = []
            base = len(self._frames)
            for i, r in enumerate(recs):
                self._frames.append((root, r))
                if r.seq_id is None:
                    iid.append(base + i)
                else:
                    by_seq.setdefault(r.seq_id, []).append(base + i)
            for seq in by_seq.values():
                seq.sort(key=lambda idx: self._frames[idx][1].seq_t)
                if len(seq) < self.window:
                    self._windows.append(seq)  # short seq: single padded window
                    continue
                starts = list(range(0, len(seq) - self.window + 1, self.stride))
                # always include a window ending at the sequence's last frame
                # (a plain stride loop drops up to stride-1 tail frames).
                if starts[-1] != len(seq) - self.window:
                    starts.append(len(seq) - self.window)
                for s in starts:
                    self._windows.append(seq[s:s + self.window])
            for idx in iid:
                self._windows.append([idx])
        if not self._windows:
            raise RuntimeError(f"no usable windows found under {roots}")

    def __len__(self) -> int:
        return len(self._windows)

    # ------------------------------------------------------------------ #
    def _load_frame(self, root: str, rec: _FrameRec):
        h, w = self.size
        img_path = os.path.join(root, "images", rec.file)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(img_path)
        H0, W0 = img.shape[:2]
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        seg_t = torch.zeros(1, h // 4, w // 4)
        if self.load_masks:
            m_path = os.path.join(root, "masks", rec.file)
            m = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                m = cv2.resize(m, (w // 4, h // 4),
                               interpolation=cv2.INTER_AREA)
                seg_t = torch.from_numpy((m > 127).astype(np.float32))[None]

        # 2-D targets normalized by ORIGINAL image size (resize-transparent).
        # PRIVILEGED SUPERVISION: corners/center are exact analytic
        # projections even when OUT OF FRAME or occluded by another gate, so
        # they are supervised whenever geometrically defined (gate in front
        # of the camera) — NOT gated on the visibility flag. The model learns
        # to infer where off-screen/occluded keypoints are; the
        # corner_inside head separately learns what is actually in frame.
        # Sanity window [-1, 2] in normalized units keeps near-90-deg-off-axis
        # tan blowups out of the loss.
        in_front = rec.depth > 0.05
        corners_uv = np.zeros((4, 2), dtype=np.float32)
        corner_ok = np.zeros(4, dtype=np.float32)
        corner_inside = np.zeros(4, dtype=np.float32)
        if rec.corners is not None and in_front:
            cn = rec.corners / np.array([W0, H0], dtype=np.float64)
            corners_uv = cn.astype(np.float32)
            sane = ((cn[:, 0] > -1.0) & (cn[:, 0] < 2.0)
                    & (cn[:, 1] > -1.0) & (cn[:, 1] < 2.0))
            corner_ok = sane.astype(np.float32)
            inside = ((rec.corners[:, 0] >= 0) & (rec.corners[:, 0] < W0)
                      & (rec.corners[:, 1] >= 0) & (rec.corners[:, 1] < H0))
            corner_inside = inside.astype(np.float32)

        center_uv = np.zeros(2, dtype=np.float32)
        center_ok = 0.0
        if rec.center_px is not None and in_front:
            cn = np.asarray(rec.center_px) / np.array([W0, H0])
            if -1.0 < cn[0] < 2.0 and -1.0 < cn[1] < 2.0:
                center_uv = cn.astype(np.float32)
                center_ok = 1.0

        dist = float(np.linalg.norm(rec.position))
        pose_ok = 1.0 if dist <= self.pose_sup_max_m else 0.0
        log_depth = float(np.log(max(rec.depth, 0.05))) if rec.depth > 0.05 \
            else 0.0
        depth_ok = 1.0 if (rec.depth > 0.05 and pose_ok > 0) else 0.0

        if rec.K is not None:
            K = np.asarray(rec.K, dtype=np.float32)
        else:
            K = np.array([[self.nominal_focal, 0, W0 / 2.0],
                          [0, self.nominal_focal, H0 / 2.0],
                          [0, 0, 1.0]], dtype=np.float32)
        k_scale = np.array([K[0, 0], K[1, 1]],
                           dtype=np.float32) / self.nominal_focal

        # ego input = [v_cam(3), omega_cam(3), dt] — dt makes the temporal
        # model RATE-AWARE (per-step motion at 120 Hz is 4x smaller than at
        # 30 Hz; without dt the GRU must guess the frame interval).
        dt = 1.0 / self._fps.get(root, 30.0)
        ego6 = (rec.ego if rec.ego is not None
                else np.zeros(6)).astype(np.float32)
        ego = np.concatenate([ego6, [np.float32(dt)]]).astype(np.float32)

        return dict(
            image=img_t, seg=seg_t,
            corners_uv=torch.from_numpy(corners_uv),
            corner_ok=torch.from_numpy(corner_ok),
            corner_inside=torch.from_numpy(corner_inside),
            center_uv=torch.from_numpy(center_uv),
            center_ok=torch.tensor(center_ok),
            position=torch.from_numpy(rec.position.astype(np.float32)),
            R=torch.from_numpy(rec.R.astype(np.float32)),
            log_depth=torch.tensor(log_depth, dtype=torch.float32),
            depth_ok=torch.tensor(depth_ok),
            pose_ok=torch.tensor(pose_ok, dtype=torch.float32),
            visible=torch.tensor(1.0 if rec.visible else 0.0),
            visible_frac=torch.tensor(rec.visible_frac, dtype=torch.float32),
            ego=torch.from_numpy(ego),
            k_scale=torch.from_numpy(k_scale),
            K=torch.from_numpy(K),
            img_wh=torch.tensor([float(W0), float(H0)]),
        )

    def __getitem__(self, i: int):
        idxs = self._windows[i]
        # Left-pad short windows by repeating the first frame: the duplicates
        # warm the recurrent state but carry frame_ok=0 (no loss / metrics)
        # and ZERO ego-motion (a repeated frame has no motion).
        n_pad = max(0, self.window - len(idxs))
        idxs = [idxs[0]] * n_pad + list(idxs)
        frames = [self._load_frame(*self._frames[j]) for j in idxs]
        out = {k: torch.stack([f[k] for f in frames], dim=0)
               for k in frames[0]}
        out["frame_ok"] = torch.cat([torch.zeros(n_pad),
                                     torch.ones(len(idxs) - n_pad)])
        if n_pad:
            out["ego"][:n_pad, :6] = 0.0  # repeated frame = no motion; dt kept
        ego_available = 1.0
        if self.ego_dropout > 0 and torch.rand(()) < self.ego_dropout:
            # modality outage drops v/omega; dt (the loop rate) is ALWAYS
            # known at deployment and survives the dropout.
            out["ego"][:, :6] = 0.0
            ego_available = 0.0
        # per-frame flag: blind-frame pose supervision is only well-posed
        # when the model can observe its own motion.
        out["ego_ok"] = torch.full((self.window,), ego_available)
        return out
