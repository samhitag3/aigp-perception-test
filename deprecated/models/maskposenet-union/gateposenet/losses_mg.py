"""Hungarian-matched multi-gate loss for GatePoseNet-MG.

Per frame, queries are assigned to GT gates by minimum-cost bipartite
matching (scipy linear_sum_assignment) over a cheap cost (presence prob,
center distance, metric-position distance). Matched pairs are supervised on
every head (masks, corners incl. privileged out-of-frame ones, center,
position, log-depth, symmetry-aware rotation, visible-fraction); unmatched
queries are pushed to presence=0. The target head is a softmax across
queries hitting the matched is_target gate.

All masking rules carry over from the single-gate loss: frame_ok kills
padded steps, blind-gate 3-D supervision is ego-gated and down-weighted,
2-D terms honor per-corner geometric validity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .losses import CORNER_PERMS, _masked_mean, gate_symmetry_group


class GatePoseMGLoss(nn.Module):
    def __init__(self, w_presence=2.0, w_mask=2.0, w_corners=5.0,
                 w_corner_in=0.5, w_center=5.0, w_position=2.0, w_depth=2.0,
                 w_rot=2.0, w_visfrac=0.5, w_target=1.0,
                 blind_scale=0.3, symmetry_aware=True,
                 far_start_m=8.0, far_full_m=20.0, far_boost=0.0,
                 small_area_frac=0.003, small_boost=0.0,
                 max_gate_weight=4.0, presence_focal_gamma=0.0):
        super().__init__()
        self.w = dict(presence=w_presence, mask=w_mask, corners=w_corners,
                      corner_in=w_corner_in, center=w_center,
                      position=w_position, depth=w_depth, rot=w_rot,
                      visfrac=w_visfrac, target=w_target)
        self.blind_scale = float(blind_scale)
        self.symmetry_aware = bool(symmetry_aware)
        self.far_start_m = float(far_start_m)
        self.far_full_m = float(far_full_m)
        self.far_boost = float(far_boost)
        self.small_area_frac = float(small_area_frac)
        self.small_boost = float(small_boost)
        self.max_gate_weight = float(max_gate_weight)
        self.presence_focal_gamma = float(presence_focal_gamma)
        self.register_buffer("sym_rots", gate_symmetry_group())
        self.register_buffer("corner_perms", CORNER_PERMS.clone())

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _match(self, pred, tgt):
        """(B, T, N_gt) -> matched query index, -1 where gate invalid."""
        B, T, Q = pred["presence_logit"].shape
        N = tgt["g_valid"].shape[2]
        prob = torch.sigmoid(pred["presence_logit"])          # (B,T,Q)
        c_cost = torch.cdist(pred["center_uv"],
                             tgt["g_center"])                 # (B,T,Q,N)
        p_cost = torch.cdist(pred["position"], tgt["g_position"]) / 5.0
        cost = (c_cost + p_cost - prob[..., None]).cpu().numpy()
        valid = tgt["g_valid"].cpu().numpy() > 0.5
        assign = torch.full((B, T, N), -1, dtype=torch.long)
        for b in range(B):
            for t in range(T):
                nv = valid[b, t].nonzero()[0]
                if len(nv) == 0:
                    continue
                r, c = linear_sum_assignment(cost[b, t][:, nv])
                for qi, gi in zip(r, c):
                    assign[b, t, nv[gi]] = qi
        return assign.to(pred["presence_logit"].device)

    # ------------------------------------------------------------------ #
    def forward(self, pred, tgt):
        B, T, Q = pred["presence_logit"].shape
        N = tgt["g_valid"].shape[2]
        frame_ok = tgt.get("frame_ok",
                           torch.ones(B, T, device=tgt["g_valid"].device))
        ego_ok = tgt.get("ego_ok", torch.ones_like(frame_ok))
        assign = self._match(pred, tgt)                       # (B,T,N)

        # gather per-gate predictions at their matched queries
        m = assign.clamp_min(0)                               # safe gather idx
        def take(key, extra_dims=0):
            v = pred[key]
            idx = m
            for _ in range(extra_dims + (v.dim() - 3)):
                idx = idx.unsqueeze(-1)
            idx = idx.expand(*m.shape, *v.shape[3:])
            return v.gather(2, idx)

        matched = (assign >= 0).float() * tgt["g_valid"] \
            * frame_ok[..., None]                             # (B,T,N)
        vis = tgt["g_visible"] * matched
        blind_w = (1 - tgt["g_visible"]) * self.blind_scale \
            * ego_ok[..., None]

        # Distance/size-aware gate weighting.  Only visible gates receive the
        # extra detection boost; invisible temporal supervision stays at 1x.
        rng = tgt["g_position"].norm(dim=-1)
        far_den = max(self.far_full_m - self.far_start_m, 1e-6)
        far_score = ((rng - self.far_start_m) / far_den).clamp(0.0, 1.0)
        gm = tgt["g_mask"]
        area_frac = gm.mean(dim=(-1, -2))
        small_den = max(self.small_area_frac, 1e-8)
        small_score = ((self.small_area_frac - area_frac) / small_den).clamp(0.0, 1.0)
        gate_weight = 1.0 + self.far_boost * far_score + self.small_boost * small_score
        gate_weight = gate_weight.clamp(1.0, self.max_gate_weight)
        gate_weight = 1.0 + (gate_weight - 1.0) * tgt["g_visible"]

        w3d = matched * (tgt["g_visible"] + blind_w) * gate_weight

        # -- presence: matched queries -> 1, all others -> 0 -------------
        pres_tgt = torch.zeros(B, T, Q, device=assign.device)
        # scatter_add is important: invalid GT slots use safe index 0 and must
        # not overwrite a real positive assignment to query 0 with zero.
        pres_tgt.scatter_add_(2, m, matched)
        pres_tgt.clamp_(0.0, 1.0)
        # Scatter the per-GT far/small boost onto its matched query.
        pres_weight = torch.ones(B, T, Q, device=assign.device)
        pres_weight.scatter_add_(2, m, (gate_weight - 1.0) * matched)
        pres_bce = F.binary_cross_entropy_with_logits(
            pred["presence_logit"], pres_tgt, reduction="none")
        if self.presence_focal_gamma > 0:
            pp = torch.sigmoid(pred["presence_logit"])
            pt = pp * pres_tgt + (1.0 - pp) * (1.0 - pres_tgt)
            pres_bce = pres_bce * (1.0 - pt).pow(self.presence_focal_gamma)
        l_presence = _masked_mean(
            pres_bce, frame_ok[..., None].expand_as(pres_tgt) * pres_weight)

        # -- instance masks (BCE + dice per matched pair) ------------------
        pm = take("mask_logit")                               # (B,T,N,h,w)
        bce = F.binary_cross_entropy_with_logits(
            pm, gm, reduction="none").mean(dim=(-1, -2))
        p = torch.sigmoid(pm)
        inter = (p * gm).sum(dim=(-1, -2))
        dice = 1 - (2 * inter + 1e-6) / (
            p.sum(dim=(-1, -2)) + gm.sum(dim=(-1, -2)) + 1e-6)
        # Do not train an empty visible-mask target for a fully occluded gate.
        # Presence/pose temporal supervision still applies to those gates.
        has_mask = (gm.sum(dim=(-1, -2)) > 0).float()
        l_mask = _masked_mean(bce + dice, matched * has_mask * gate_weight)

        # -- symmetry element per gate (rotation-min; corners follow) -----
        Rp = take("R")                                        # (B,T,N,3,3)
        Rg = torch.einsum("btnij,gjk->btngik", tgt["g_R"], self.sym_rots)
        d_rot = ((Rp[:, :, :, None] - Rg) ** 2).sum(dim=(-1, -2)) / 4.0
        g_best = d_rot.argmin(dim=-1)                         # (B,T,N)
        l_rot = _masked_mean(
            d_rot.gather(3, g_best[..., None]).squeeze(-1), w3d)

        # -- corners (privileged; permuted by the chosen symmetry) --------
        perm = self.corner_perms.to(assign.device)[g_best]    # (B,T,N,4)
        tc = tgt["g_corners"].gather(
            3, perm[..., None].expand(*perm.shape, 2))
        ti = tgt["g_inside"].gather(3, perm)
        tok = tgt["g_corner_ok"].gather(3, perm)
        pc = take("corners_uv")
        l_corners = _masked_mean(
            F.smooth_l1_loss(pc, tc, reduction="none", beta=0.02).sum(-1),
            tok * w3d[..., None])
        l_corner_in = _masked_mean(
            F.binary_cross_entropy_with_logits(
                take("corner_inside_logit"), ti, reduction="none"),
            tok * matched[..., None])

        # -- center / position / depth / visfrac --------------------------
        l_center = _masked_mean(
            F.smooth_l1_loss(take("center_uv"), tgt["g_center"],
                             reduction="none", beta=0.02).sum(-1),
            tgt["g_center_ok"] * w3d)
        l_position = _masked_mean(
            F.smooth_l1_loss(take("position"), tgt["g_position"],
                             reduction="none", beta=0.1).sum(-1)
            / (1.0 + rng), w3d)
        l_depth = _masked_mean(
            F.smooth_l1_loss(take("log_depth"), tgt["g_log_depth"],
                             reduction="none", beta=0.05),
            tgt["g_depth_ok"] * matched)
        l_visfrac = _masked_mean(
            F.smooth_l1_loss(take("visible_frac"), tgt["g_visible_frac"],
                             reduction="none", beta=0.05), matched)

        # -- target: softmax over queries must pick the is_target gate ----
        t_logits = pred["target_logit"]                       # (B,T,Q)
        tgt_q = torch.full((B, T), -100, dtype=torch.long,
                           device=assign.device)
        has_t = (tgt["g_target"] * matched).max(dim=2)
        gate_idx = has_t.indices                              # (B,T)
        q_idx = assign.gather(2, gate_idx[..., None]).squeeze(-1)
        sel = (has_t.values > 0.5) & (q_idx >= 0) & (frame_ok > 0.5)
        if sel.any():
            l_target = F.cross_entropy(
                t_logits[sel], q_idx[sel])
        else:
            l_target = t_logits.sum() * 0.0

        total = sum(self.w[k] * v for k, v in [
            ("presence", l_presence), ("mask", l_mask),
            ("corners", l_corners), ("corner_in", l_corner_in),
            ("center", l_center), ("position", l_position),
            ("depth", l_depth), ("rot", l_rot), ("visfrac", l_visfrac),
            ("target", l_target)])
        logs = {k: float(v.detach()) for k, v in [
            ("presence", l_presence), ("mask", l_mask),
            ("corners", l_corners), ("corner_in", l_corner_in),
            ("center", l_center), ("position", l_position),
            ("depth", l_depth), ("rot", l_rot), ("visfrac", l_visfrac),
            ("target", l_target), ("total", total)]}
        return total, logs
