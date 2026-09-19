"""Multi-task loss for GatePoseNet.

Every term is masked by the corresponding GT validity mask from the dataset —
the central rule is *never supervise a target that is not defined for the
frame* (e.g. 2-D corners while the gate is behind the camera), and *do*
supervise the 3-D pose through blind frames (that is what teaches the
recurrent state + ego-motion to track the gate while it is out of view).

Terms (weights in the config `loss:` block):
  seg          BCE + Dice on the 1/4-res aux mask (always defined)
  corners      smooth-L1 on normalized corner coords (visible frames)
  corner_in    BCE per-corner inside-frame classification (visible frames)
  center       smooth-L1 on the normalized fly-through center (visible+front)
  position     smooth-L1 on position_cam, scaled 1/(1+range) => ~relative error
  depth        smooth-L1 on log(center_depth)
  rot          chordal loss ||R_pred - R_gt||_F^2 / 4 (in [0,1]-ish range);
               invisible frames get `rot_blind_scale` weight (memory shaping)
  vis          BCE on the visibility flag + smooth-L1 on visible_frac
  kcal         smooth-L1 on (fx,fy)/nominal (auxiliary self-calibration)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gate_symmetry_group() -> torch.Tensor:
    """(8, 3, 3) proper rotations of the D4 symmetry of a square gate ring.

    In the gate body frame (+X right, +Y down, +Z fly-through): the 4
    rotations about +Z by k*90 deg, and each composed with a 180-deg rotation
    about +X (front/back flip — a plain square ring looks identical from
    behind). Over full-attitude / full-sphere training data the gate's
    orientation is only identifiable modulo this group, so supervising the raw
    world-derived R would inject label ambiguity; the loss instead scores the
    best-matching symmetry element.
    """
    mats = []
    Rx180 = torch.tensor([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    for k in range(4):
        a = math.pi / 2 * k
        Rz = torch.tensor([
            [math.cos(a), -math.sin(a), 0.0],
            [math.sin(a), math.cos(a), 0.0],
            [0.0, 0.0, 1.0],
        ])
        mats.append(Rz)
        mats.append(Rz @ Rx180)
    return torch.stack(mats, dim=0)


def _corner_perms_for(rots: torch.Tensor) -> torch.Tensor:
    """(8, 4) corner-label permutations ROW-ALIGNED with ``rots``.

    Gate corners in the body frame (+X right, +Y down): TL=(-a,-a,0),
    TR=(a,-a,0), BR=(a,a,0), BL=(-a,a,0). Relabeling the body frame by
    symmetry g (R' = R @ g) puts new-label i at old-frame position g @ c_i,
    so perm[g][i] = j with c_j == g @ c_i — derived here rather than
    hand-written so row g of the permutations always corresponds to row g of
    the rotation group (a hand-written table previously matched only as a
    set, which broke any coupled use of the two).
    """
    c = torch.tensor([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0],
                      [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
    perms = []
    for g in rots:
        mapped = c @ g.T  # position of new-label i in the old frame
        row = []
        for i in range(4):
            d = (c - mapped[i]).norm(dim=-1)
            j = int(d.argmin())
            assert float(d[j]) < 1e-6, "symmetry element is not a square symmetry"
            row.append(j)
        assert sorted(row) == [0, 1, 2, 3]
        perms.append(row)
    return torch.tensor(perms, dtype=torch.long)


CORNER_PERMS = _corner_perms_for(gate_symmetry_group())


def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Mean of x over elements where mask m > 0 (broadcast; safe when empty)."""
    while m.dim() < x.dim():
        m = m.unsqueeze(-1)
    m = m.expand_as(x).float()
    denom = m.sum().clamp_min(1.0)
    return (x * m).sum() / denom


class GatePoseLoss(nn.Module):
    def __init__(
        self,
        w_seg: float = 1.0,
        w_corners: float = 5.0,
        w_corner_in: float = 0.5,
        w_center: float = 5.0,
        w_position: float = 2.0,
        w_depth: float = 2.0,
        w_rot: float = 2.0,
        w_vis: float = 1.0,
        w_kcal: float = 0.2,
        rot_blind_scale: float = 0.3,
        pos_blind_scale: float = 0.3,
        symmetry_aware: bool = True,
    ):
        super().__init__()
        self.w = dict(seg=w_seg, corners=w_corners, corner_in=w_corner_in,
                      center=w_center, position=w_position, depth=w_depth,
                      rot=w_rot, vis=w_vis, kcal=w_kcal)
        self.rot_blind_scale = float(rot_blind_scale)
        self.pos_blind_scale = float(pos_blind_scale)
        self.symmetry_aware = bool(symmetry_aware)
        self.register_buffer("sym_rots", gate_symmetry_group())
        self.register_buffer("corner_perms", CORNER_PERMS.clone())

    def forward(self, pred: dict, tgt: dict):
        logs: dict[str, float] = {}
        B, T = tgt["visible"].shape[:2]
        # frame_ok: 0 for left-padding duplicates in short windows — those
        # steps warm the recurrent state but must not contribute loss.
        frame_ok = tgt.get("frame_ok",
                           torch.ones_like(tgt["visible"]))
        vis = tgt["visible"] * frame_ok          # (B,T) 1 when gate visible
        pose_ok = tgt["pose_ok"] * frame_ok      # (B,T)
        # ego_ok: 0 when the ego-motion modality was dropped for the window —
        # blind-frame pose is then unobservable, so its supervision is off.
        ego_ok = tgt.get("ego_ok", torch.ones_like(frame_ok))

        # -- aux segmentation (per-frame, masked by frame_ok) -----------------
        seg_logit = pred["seg_logit"]                    # (B,T,1,h,w)
        seg_tgt = tgt["seg"]
        bce = F.binary_cross_entropy_with_logits(
            seg_logit, seg_tgt, reduction="none").mean(dim=(2, 3, 4))
        p = torch.sigmoid(seg_logit)
        inter = (p * seg_tgt).sum(dim=(2, 3, 4))
        dice = 1 - (2 * inter + 1e-6) / (
            p.sum(dim=(2, 3, 4)) + seg_tgt.sum(dim=(2, 3, 4)) + 1e-6)
        l_seg = _masked_mean(bce + dice, frame_ok)

        # -- symmetry element selection (ONE g per frame, shared by the
        # rotation AND corner terms so the two heads stay mutually
        # consistent). Chosen by rotation distance, which is defined for
        # every frame; the corner labels follow via the row-aligned perms.
        if self.symmetry_aware:
            Rg = torch.einsum("btij,gjk->btgik", tgt["R"], self.sym_rots)
            d_rot = ((pred["R"][:, :, None] - Rg) ** 2).sum(dim=(-1, -2)) / 4.0
            g_best = d_rot.argmin(dim=-1)                 # (B,T)

        # -- 2-D geometry: PRIVILEGED supervision — corners are exact even
        # when out of frame / occluded (corner_ok encodes geometric validity,
        # not visibility). Off-screen keypoints of a non-visible gate are
        # inferable only via temporal state + ego-motion, so they get the
        # blind-frame weight; visible frames get full weight.
        blind_w = (1 - vis) * self.pos_blind_scale * ego_ok
        c_mask = tgt["corner_ok"] * (vis + blind_w)[..., None]   # (B,T,4)
        if self.symmetry_aware:
            perm = self.corner_perms[g_best]              # (B,T,4)
            tc = tgt["corners_uv"].gather(
                2, perm[..., None].expand(-1, -1, -1, 2))  # (B,T,4,2)
            ti = tgt["corner_inside"].gather(2, perm)      # (B,T,4)
            l_corners = _masked_mean(
                F.smooth_l1_loss(pred["corners_uv"], tc,
                                 reduction="none", beta=0.02).sum(-1),
                c_mask)
            l_corner_in = _masked_mean(
                F.binary_cross_entropy_with_logits(
                    pred["corner_inside_logit"], ti, reduction="none"),
                tgt["corner_ok"] * frame_ok[..., None])
        else:
            l_corners = _masked_mean(
                F.smooth_l1_loss(pred["corners_uv"], tgt["corners_uv"],
                                 reduction="none", beta=0.02).sum(-1),
                c_mask)
            l_corner_in = _masked_mean(
                F.binary_cross_entropy_with_logits(
                    pred["corner_inside_logit"], tgt["corner_inside"],
                    reduction="none"),
                tgt["corner_ok"] * frame_ok[..., None])
        l_center = _masked_mean(
            F.smooth_l1_loss(pred["center_uv"], tgt["center_uv"],
                             reduction="none", beta=0.02).sum(-1),
            tgt["center_ok"] * (vis + blind_w))

        # -- 3-D pose ---------------------------------------------------------
        # Blind-frame supervision requires ego-motion to be observable: when
        # the modality was dropped for the window, blind weights go to zero.
        rng = tgt["position"].norm(dim=-1)               # (B,T) true range
        pos_scale = 1.0 / (1.0 + rng)
        pos_w = pose_ok * (vis + (1 - vis) * self.pos_blind_scale * ego_ok)
        l_position = _masked_mean(
            F.smooth_l1_loss(pred["position"], tgt["position"],
                             reduction="none", beta=0.1).sum(-1) * pos_scale,
            pos_w)
        l_depth = _masked_mean(
            F.smooth_l1_loss(pred["log_depth"], tgt["log_depth"],
                             reduction="none", beta=0.05),
            tgt["depth_ok"] * frame_ok)
        rot_w = pose_ok * (vis + (1 - vis) * self.rot_blind_scale * ego_ok)
        if self.symmetry_aware:
            l_rot = _masked_mean(
                d_rot.gather(2, g_best[..., None]).squeeze(-1), rot_w)
        else:
            l_rot = _masked_mean(
                ((pred["R"] - tgt["R"]) ** 2).sum(dim=(-1, -2)) / 4.0, rot_w)

        # -- visibility + self-calibration ------------------------------------
        l_vis = (_masked_mean(
                    F.binary_cross_entropy_with_logits(
                        pred["visible_logit"], tgt["visible"],
                        reduction="none"), frame_ok)
                 + _masked_mean(
                    F.smooth_l1_loss(pred["visible_frac"],
                                     tgt["visible_frac"],
                                     reduction="none", beta=0.05), frame_ok))
        l_kcal = _masked_mean(
            F.smooth_l1_loss(pred["k_scale"], tgt["k_scale"],
                             reduction="none", beta=0.01), frame_ok)

        total = (self.w["seg"] * l_seg
                 + self.w["corners"] * l_corners
                 + self.w["corner_in"] * l_corner_in
                 + self.w["center"] * l_center
                 + self.w["position"] * l_position
                 + self.w["depth"] * l_depth
                 + self.w["rot"] * l_rot
                 + self.w["vis"] * l_vis
                 + self.w["kcal"] * l_kcal)
        for k, v in [("seg", l_seg), ("corners", l_corners),
                     ("corner_in", l_corner_in), ("center", l_center),
                     ("position", l_position), ("depth", l_depth),
                     ("rot", l_rot), ("vis", l_vis), ("kcal", l_kcal),
                     ("total", total)]:
            logs[k] = float(v.detach())
        return total, logs
