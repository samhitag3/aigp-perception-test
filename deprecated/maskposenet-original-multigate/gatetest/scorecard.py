"""Standardized edge-case scorecard for gate-pose models.

One number per cell of a FIXED slice grid, so runs/models/datasets are
directly comparable. Slices (rows) x metrics (columns):

Slices
------
* overall
* kind:<flythrough|hover|passby|corkscrew|loop|staggered>   (maneuver)
* vis:visible / vis:blind                                    (target gate)
* occ:full (vf>0.95) / occ:partial (0.3-0.95) / occ:sliver (<0.3, visible)
* att:upright (|roll|<45) / att:banked (45-135) / att:inverted (>135)
* range:0-3m / 3-6m / 6-10m / 10-25m
* scene:single / scene:multi (2+ gates)
* switch:±5 frames around a target switch                    (maneuvers)

Metrics per slice: n frames, position error m (mean/median), relative
position error (median), fly-through-axis error deg (median), center px
(median, visible only), visibility accuracy.

The model runs SEQUENTIALLY per sequence through ``model.step`` (the real
deployment loop, hidden state carried), not on shuffled windows.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gateposenet.losses import gate_symmetry_group  # noqa: E402

RANGE_BUCKETS = ((0, 3), (3, 6), (6, 10), (10, 25))


def _cam_roll_deg(extrinsic) -> float:
    R = np.asarray(extrinsic, dtype=np.float64)[:3, :3]
    u = R @ np.array([0.0, 0.0, 1.0])
    return float(np.degrees(np.arctan2(-u[0], -u[1])))


def _load_frames(root: str):
    with open(os.path.join(root, "ground_truth.json")) as f:
        gt = json.load(f)
    fps = float(gt.get("fps") or 30.0)
    frames: dict = {}
    n_gates: dict = {}
    for o in gt.get("objects", []):
        f_id = o["frame"]
        n_gates[f_id] = n_gates.get(f_id, 0) + 1
        is_t = o.get("is_target")
        if (is_t is True) or (is_t is None and int(o.get("obj_id", 0)) == 0):
            frames[f_id] = o
    seqs: dict = {}
    for f_id, o in sorted(frames.items()):
        seqs.setdefault(o.get("seq_id", f"iid{f_id}"), []).append(o)
    for s in seqs.values():
        s.sort(key=lambda o: (o.get("seq_t") or 0))
    return seqs, n_gates, fps


@torch.no_grad()
def run_model_on_dataset(model, root: str, device, size=(192, 320),
                         fps_override: float | None = None):
    """Stream every sequence through model.step; returns per-frame records."""
    seqs, n_gates, fps = _load_frames(root)
    if fps_override:
        fps = fps_override
    G = gate_symmetry_group().numpy()
    recs = []
    for seq_id, objs in seqs.items():
        h = None
        prev_target = None
        switch_countdown = -99
        for o in objs:
            img = cv2.imread(os.path.join(root, "images", o["file"]))
            if img is None:
                continue
            H0, W0 = img.shape[:2]
            rgb = cv2.cvtColor(cv2.resize(img, (size[1], size[0]),
                                          interpolation=cv2.INTER_AREA),
                               cv2.COLOR_BGR2RGB)
            x = (torch.from_numpy(rgb.transpose(2, 0, 1))[None].float()
                 / 255.0).to(device)
            ego6 = o.get("ego_motion_cam") or [0.0] * 6
            ego = torch.tensor([[*ego6, 1.0 / fps]], dtype=torch.float32,
                               device=device)
            out, h = model.step(x, ego, h)

            tgt_obj = int(o.get("obj_id", 0))
            if prev_target is not None and tgt_obj != prev_target:
                switch_countdown = 5
            prev_target = tgt_obj

            p_gt = np.asarray(o["position_cam"])
            p_pred = out["position"][0].cpu().numpy()
            R_gt = np.asarray(o["R_cam_gate"])
            R_pred = out["R"][0].cpu().numpy()
            # symmetry-aware axis + rotation errors
            axes = np.stack([(R_gt @ g)[:, 2] for g in G])
            a_pred = R_pred[:, 2]
            axis_err = float(np.degrees(np.arccos(np.clip(
                np.abs(axes @ a_pred).max(), -1, 1))))
            vis_p = float(torch.sigmoid(out["visible_logit"])[0])
            center_err = None
            if o.get("visible") and o.get("center_px") is not None \
                    and (o.get("center_depth_m") or 0) > 0.05:
                c_pred = out["center_uv"][0].cpu().numpy() * [W0, H0]
                center_err = float(np.linalg.norm(
                    c_pred - np.asarray(o["center_px"])))
            recs.append(dict(
                seq_id=seq_id, kind=o.get("seq_kind") or "iid",
                visible=bool(o.get("visible")),
                visible_frac=float(o.get("visible_frac") or 0.0),
                roll=abs(_cam_roll_deg(o["extrinsic"])),
                rng=float(np.linalg.norm(p_gt)),
                gates=n_gates.get(o["frame"], 1),
                near_switch=switch_countdown >= 0,
                pos_err=float(np.linalg.norm(p_pred - p_gt)),
                axis_err=axis_err,
                center_err=center_err,
                vis_correct=(vis_p >= 0.5) == bool(o.get("visible")),
            ))
            switch_countdown -= 1
    return recs


def _slice_masks(recs):
    def m(fn):
        return [i for i, r in enumerate(recs) if fn(r)]
    kinds = sorted({r["kind"] for r in recs})
    slices = {"overall": m(lambda r: True)}
    for k in kinds:
        slices[f"kind:{k}"] = m(lambda r, k=k: r["kind"] == k)
    slices["vis:visible"] = m(lambda r: r["visible"])
    slices["vis:blind"] = m(lambda r: not r["visible"])
    slices["occ:full"] = m(lambda r: r["visible"] and r["visible_frac"] > 0.95)
    slices["occ:partial"] = m(
        lambda r: r["visible"] and 0.3 <= r["visible_frac"] <= 0.95)
    slices["occ:sliver"] = m(
        lambda r: r["visible"] and r["visible_frac"] < 0.3)
    slices["att:upright"] = m(lambda r: r["roll"] < 45)
    slices["att:banked"] = m(lambda r: 45 <= r["roll"] <= 135)
    slices["att:inverted"] = m(lambda r: r["roll"] > 135)
    for lo, hi in RANGE_BUCKETS:
        slices[f"range:{lo}-{hi}m"] = m(
            lambda r, lo=lo, hi=hi: lo <= r["rng"] < hi)
    slices["scene:single"] = m(lambda r: r["gates"] == 1)
    slices["scene:multi"] = m(lambda r: r["gates"] >= 2)
    slices["switch:±5frames"] = m(lambda r: r["near_switch"])
    return slices


def build_scorecard(recs) -> dict:
    slices = _slice_masks(recs)
    card = {}
    for name, idx in slices.items():
        if not idx:
            card[name] = {"n": 0}
            continue
        sub = [recs[i] for i in idx]
        pos = np.array([r["pos_err"] for r in sub])
        rel = np.array([r["pos_err"] / max(r["rng"], 1e-6) for r in sub])
        axi = np.array([r["axis_err"] for r in sub])
        cen = np.array([r["center_err"] for r in sub
                        if r["center_err"] is not None])
        card[name] = {
            "n": len(sub),
            "pos_err_m_mean": round(float(pos.mean()), 3),
            "pos_err_m_median": round(float(np.median(pos)), 3),
            "pos_err_rel_median": round(float(np.median(rel)), 4),
            "axis_err_deg_median": round(float(np.median(axi)), 2),
            "center_err_px_median": (round(float(np.median(cen)), 1)
                                     if len(cen) else None),
            "vis_acc": round(float(np.mean([r["vis_correct"]
                                            for r in sub])), 3),
        }
    return card


def format_scorecard_md(card: dict, title: str) -> str:
    lines = [f"# Gate-pose scorecard — {title}", "",
             "| slice | n | pos err m (mean/med) | rel err (med) | "
             "axis° (med) | center px (med) | vis acc |",
             "|---|---|---|---|---|---|---|"]
    for name, v in card.items():
        if v.get("n", 0) == 0:
            lines.append(f"| {name} | 0 | — | — | — | — | — |")
            continue
        c = v.get("center_err_px_median")
        lines.append(
            f"| {name} | {v['n']} | {v['pos_err_m_mean']} / "
            f"{v['pos_err_m_median']} | {v['pos_err_rel_median']} | "
            f"{v['axis_err_deg_median']} | {c if c is not None else '—'} | "
            f"{v['vis_acc']} |")
    return "\n".join(lines) + "\n"
