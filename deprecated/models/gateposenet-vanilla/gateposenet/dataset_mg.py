"""Multi-gate rolling-window dataset: PER-GATE targets, padded to N_MAX.

Same window mechanics as GateSequenceDataset, but every frame carries ALL
gates in the scene (the generator writes per-gate masks
``masks/frame_XXXXX_obj_<i>.png`` and one GT record per gate):

    image        (T, 3, H, W)
    ego          (T, 7)                    [v, omega, dt]
    frame_ok/ego_ok (T,)
    gates:
      g_valid      (T, N) 1 = real gate record (else padding)
      g_mask       (T, N, H/4, W/4)        per-gate instance masks
      g_corners    (T, N, 4, 2) + g_corner_ok (T, N, 4) + g_inside (T, N, 4)
      g_center     (T, N, 2) + g_center_ok (T, N)
      g_position   (T, N, 3), g_R (T, N, 3, 3), g_log_depth/depth_ok (T, N)
      g_visible    (T, N), g_visible_frac (T, N)
      g_target     (T, N) one-hot-ish: the gate being flown at
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

N_MAX = 4


class GateSequenceDatasetMG(Dataset):
    def __init__(self, roots, window=8, stride=4, size=(192, 320),
                 pose_sup_max_m=25.0, nominal_focal=320.0,
                 ego_dropout=0.0):
        if isinstance(roots, str):
            roots = [roots]
        self.size = (int(size[0]), int(size[1]))
        self.window = int(window)
        self.stride = max(1, int(stride))
        self.pose_sup_max_m = float(pose_sup_max_m)
        self.nominal_focal = float(nominal_focal)
        self.ego_dropout = float(ego_dropout)

        self._frames: list[tuple[str, dict, float]] = []  # (root, frame, fps)
        self._windows: list[list[int]] = []
        for root in roots:
            with open(os.path.join(root, "ground_truth.json")) as f:
                gt = json.load(f)
            fps = float(gt.get("fps") or 30.0)
            by_frame: dict = {}
            for o in gt.get("objects", []):
                by_frame.setdefault(o["frame"], []).append(o)
            by_seq: dict = {}
            base = len(self._frames)
            items = sorted(by_frame.items())
            for i, (fid, objs) in enumerate(items):
                self._frames.append((root, {"objs": objs}, fps))
                sid = objs[0].get("seq_id")
                if sid is not None:
                    by_seq.setdefault(sid, []).append(base + i)
            for seq in by_seq.values():
                seq.sort(key=lambda idx:
                         (self._frames[idx][1]["objs"][0].get("seq_t") or 0))
                if len(seq) < self.window:
                    self._windows.append(seq)
                    continue
                starts = list(range(0, len(seq) - self.window + 1,
                                    self.stride))
                if starts[-1] != len(seq) - self.window:
                    starts.append(len(seq) - self.window)
                for s in starts:
                    self._windows.append(seq[s:s + self.window])
        if not self._windows:
            raise RuntimeError(f"no windows under {roots}")

    def __len__(self):
        return len(self._windows)

    # ------------------------------------------------------------------ #
    def _load_frame(self, root, fr, fps):
        h, w = self.size
        objs = fr["objs"][:N_MAX]
        o0 = objs[0]
        img = cv2.imread(os.path.join(root, "images", o0["file"]),
                         cv2.IMREAD_COLOR)
        H0, W0 = img.shape[:2]
        img_t = torch.from_numpy(cv2.cvtColor(
            cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB).transpose(2, 0, 1)).float() / 255.0

        g = dict(
            g_valid=torch.zeros(N_MAX),
            g_mask=torch.zeros(N_MAX, h // 4, w // 4),
            g_corners=torch.zeros(N_MAX, 4, 2),
            g_corner_ok=torch.zeros(N_MAX, 4),
            g_inside=torch.zeros(N_MAX, 4),
            g_center=torch.zeros(N_MAX, 2),
            g_center_ok=torch.zeros(N_MAX),
            g_position=torch.zeros(N_MAX, 3),
            g_R=torch.eye(3).repeat(N_MAX, 1, 1),
            g_log_depth=torch.zeros(N_MAX),
            g_depth_ok=torch.zeros(N_MAX),
            g_visible=torch.zeros(N_MAX),
            g_visible_frac=torch.zeros(N_MAX),
            g_target=torch.zeros(N_MAX),
        )
        stem = o0["file"].rsplit(".", 1)[0]
        for i, o in enumerate(objs):
            if o.get("position_cam") is None:
                continue
            dist = float(np.linalg.norm(o["position_cam"]))
            if dist > self.pose_sup_max_m:
                continue
            g["g_valid"][i] = 1.0
            mp = os.path.join(root, "masks", f"{stem}_obj_{i}.png")
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if m is None and i == 0:
                m = cv2.imread(os.path.join(root, "masks", f"{stem}.png"),
                               cv2.IMREAD_GRAYSCALE)
            if m is not None:
                m = cv2.resize(m, (w // 4, h // 4),
                               interpolation=cv2.INTER_AREA)
                g["g_mask"][i] = torch.from_numpy(
                    (m > 127).astype(np.float32))
            depth = float(o.get("center_depth_m") or 0.0)
            in_front = depth > 0.05
            if o.get("corners") is not None and in_front:
                c = np.asarray(o["corners"]) / [W0, H0]
                g["g_corners"][i] = torch.from_numpy(c.astype(np.float32))
                sane = ((c[:, 0] > -1) & (c[:, 0] < 2)
                        & (c[:, 1] > -1) & (c[:, 1] < 2))
                g["g_corner_ok"][i] = torch.from_numpy(
                    sane.astype(np.float32))
                cc = np.asarray(o["corners"])
                inside = ((cc[:, 0] >= 0) & (cc[:, 0] < W0)
                          & (cc[:, 1] >= 0) & (cc[:, 1] < H0))
                g["g_inside"][i] = torch.from_numpy(
                    inside.astype(np.float32))
            if o.get("center_px") is not None and in_front:
                cn = np.asarray(o["center_px"]) / [W0, H0]
                if -1 < cn[0] < 2 and -1 < cn[1] < 2:
                    g["g_center"][i] = torch.from_numpy(
                        cn.astype(np.float32))
                    g["g_center_ok"][i] = 1.0
            g["g_position"][i] = torch.tensor(o["position_cam"],
                                              dtype=torch.float32)
            g["g_R"][i] = torch.tensor(o["R_cam_gate"], dtype=torch.float32)
            if in_front:
                g["g_log_depth"][i] = float(np.log(max(depth, 0.05)))
                g["g_depth_ok"][i] = 1.0
            g["g_visible"][i] = 1.0 if o.get("visible") else 0.0
            g["g_visible_frac"][i] = float(o.get("visible_frac") or 0.0)
            g["g_target"][i] = 1.0 if o.get("is_target") else 0.0
        # legacy single-gate data without is_target: gate 0 is the target
        if g["g_target"].sum() == 0 and g["g_valid"][0] > 0:
            g["g_target"][0] = 1.0

        ego6 = (np.asarray(o0.get("ego_motion_cam") or [0.0] * 6,
                           dtype=np.float32))
        ego = np.concatenate([ego6, [np.float32(1.0 / fps)]])
        return dict(image=img_t, ego=torch.from_numpy(ego), **g)

    def __getitem__(self, i):
        idxs = self._windows[i]
        n_pad = max(0, self.window - len(idxs))
        idxs = [idxs[0]] * n_pad + list(idxs)
        frames = [self._load_frame(*self._frames[j]) for j in idxs]
        out = {k: torch.stack([f[k] for f in frames]) for k in frames[0]}
        out["frame_ok"] = torch.cat([torch.zeros(n_pad),
                                     torch.ones(len(idxs) - n_pad)])
        if n_pad:
            out["ego"][:n_pad, :6] = 0.0
        ego_available = 1.0
        if self.ego_dropout > 0 and torch.rand(()) < self.ego_dropout:
            out["ego"][:, :6] = 0.0
            ego_available = 0.0
        out["ego_ok"] = torch.full((self.window,), ego_available)
        return out
