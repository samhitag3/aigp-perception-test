#!/usr/bin/env python3
"""ALL-gate tracking + live telemetry + side-by-side localization map.

The right presentation of what the system actually knows, per frame:

* LEFT — the camera view with the model's segmentation of EVERY gate
  (colored mask overlay per tracked gate), 4 keypoints per gate (QuAdGate
  corner extraction on each mask component), and the temporal TARGET head
  drawn as EMA-smoothed keypoint markers (no vibrating quad; the rigid
  snapped outline appears only when the PnP check passes).
* RIGHT — an offline localization map built while flying: top-down
  camera-frame view (drone at origin, 90° FOV wedge, range rings) with every
  tracked gate's PnP position, persistent per-track trails, the target
  head's position as a star, and a live telemetry column (distance,
  visibility, residual, agreement, gates in track, frame jump).

Tracks are greedy nearest-neighbor associations over the per-gate PnP
positions (camera frame), giving stable per-gate identities and colors.

    uv run python scripts/track_and_map.py --frames <dir> [--fps 30]
    uv run python scripts/track_and_map.py --dataset <synthetic dir>  # uses images/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatenet.utils import load_checkpoint, load_config, pick_device
from gateposenet.model_single import \
    build_gateposenet_single as build_gateposenet
from perception.geometric_filter import filter_gate_estimate
from perception.pose import quad_from_mask

GATE_M = 2.7
TRACK_COLORS = [(255, 255, 0), (0, 255, 128), (255, 0, 255),
                (0, 200, 255), (128, 255, 0), (255, 128, 255)]  # cyan/green/magenta family: high contrast on red-orange gates


def pnp_pos(quad, K):
    h = GATE_M / 2.0
    obj = np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]], float)
    ok, _, tvec = cv2.solvePnP(obj, quad.astype(np.float64), K, None,
                               flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    t = tvec.reshape(3)
    return t if 0.3 < t[2] < 40 else None


class Tracks:
    """Greedy NN association of per-gate camera-frame positions."""

    def __init__(self, gate_dist=2.0, max_age=20):
        self.tracks: dict = {}
        self.next_id = 0
        self.gate_dist = gate_dist
        self.max_age = max_age

    def update(self, detections):
        assigned = {}
        used = set()
        for tid, tr in sorted(self.tracks.items(),
                              key=lambda kv: kv[1]["age"]):
            best, bd = None, self.gate_dist
            for i, p in enumerate(detections):
                if i in used:
                    continue
                d = np.linalg.norm(p - tr["pos"])
                if d < bd:
                    best, bd = i, d
            if best is not None:
                used.add(best)
                tr["pos"] = detections[best]
                tr["age"] = 0
                tr["trail"].append(detections[best].copy())
                tr["trail"] = tr["trail"][-90:]
                assigned[best] = tid
        for i, p in enumerate(detections):
            if i not in used:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"pos": p.copy(), "age": 0,
                                    "trail": [p.copy()]}
                assigned[i] = tid
        dead = []
        for tid, tr in self.tracks.items():
            if tid not in assigned.values():
                tr["age"] += 1
                if tr["age"] > self.max_age:
                    dead.append(tid)
        for tid in dead:
            del self.tracks[tid]
        return assigned


def draw_map(panel_h, panel_w, tracks, target_pos, vis, telemetry):
    m = np.full((panel_h, panel_w, 3), 24, np.uint8)
    cx, cy = panel_w // 2, panel_h - 40
    scale = (panel_h - 90) / 16.0            # 16 m forward -> panel top

    def to_px(p):
        return (int(cx + p[0] * scale), int(cy - p[2] * scale))

    # FOV wedge (90 deg) + range rings
    for r in (2.5, 5, 10, 15):
        cv2.circle(m, (cx, cy), int(r * scale), (55, 55, 55), 1)
        cv2.putText(m, f"{r:g}m", (cx + 4, cy - int(r * scale) + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (110, 110, 110), 1,
                    cv2.LINE_AA)
    for s in (-1, 1):
        cv2.line(m, (cx, cy), (int(cx + s * panel_h * 0.9),
                               int(cy - panel_h * 0.9)), (60, 60, 60), 1)
    cv2.drawMarker(m, (cx, cy), (255, 255, 255), cv2.MARKER_TRIANGLE_UP,
                   14, 2)

    for tid, tr in tracks.tracks.items():
        col = TRACK_COLORS[tid % len(TRACK_COLORS)]
        for i, p in enumerate(tr["trail"][:-1]):
            a = 0.15 + 0.85 * i / max(1, len(tr["trail"]) - 1)
            c = tuple(int(v * a) for v in col)
            cv2.circle(m, to_px(p), 1, c, -1)
        if tr["age"] == 0:
            cv2.rectangle(m, tuple(np.subtract(to_px(tr["pos"]), 4)),
                          tuple(np.add(to_px(tr["pos"]), 4)), col, 2)
            cv2.putText(m, f"G{tid}", np.add(to_px(tr["pos"]), (6, 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

    if target_pos is not None:
        p = to_px(target_pos)
        col = (0, 255, 255) if vis >= 0.5 else (0, 120, 255)
        cv2.drawMarker(m, p, col, cv2.MARKER_STAR, 16, 2)

    y = 18
    for line in telemetry:
        cv2.putText(m, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 255, 255), 1, cv2.LINE_AA)
        y += 17
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gateposenet_traj.yaml")
    ap.add_argument("--checkpoint", default="runs/gateposenet_traj/best.pt")
    ap.add_argument("--frames", default=None)
    ap.add_argument("--dataset", default=None,
                    help="synthetic dataset dir (uses its images/)")
    ap.add_argument("--out", default="runs/track_and_map")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--fx", type=float, default=320.0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    frames_dir = Path(args.frames) if args.frames else \
        Path(args.dataset) / "images"
    paths = sorted(p for p in frames_dir.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if args.max_frames:
        paths = paths[:args.max_frames]
    if not paths:
        raise SystemExit(f"no frames in {frames_dir}")

    cfg = load_config(args.config)
    device = pick_device(args.device)
    d = cfg["data"]
    mh, mw = int(d.get("height", 192)), int(d.get("width", 320))
    model = build_gateposenet(cfg["model"]).to(device).eval()
    load_checkpoint(args.checkpoint, model, map_location=device)

    first = cv2.imread(str(paths[0]))
    H0, W0 = first.shape[:2]
    VW, VH = 960, 540                              # left panel
    MPW = 420                                      # map panel width
    K = np.array([[args.fx, 0, W0 / 2.0], [0, args.fx, H0 / 2.0],
                  [0, 0, 1.0]])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_dir / "track_map.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                         (VW + MPW, VH))

    ego = torch.zeros(1, 7, device=device)
    ego[0, 6] = 1.0 / args.fps
    h = None
    tracks = Tracks()
    ema_kp = None
    prev_tpos = None
    with torch.no_grad():
        for fi, p in enumerate(paths):
            img = cv2.imread(str(p))
            if img is None:
                continue
            rgb = cv2.cvtColor(cv2.resize(img, (mw, mh),
                                          interpolation=cv2.INTER_AREA),
                               cv2.COLOR_BGR2RGB)
            x = (torch.from_numpy(rgb.transpose(2, 0, 1))[None].float()
                 / 255.0).to(device)
            out, h = model.step(x, ego, h)

            vis = float(torch.sigmoid(out["visible_logit"])[0])
            t_pos = out["position"][0].cpu().numpy()
            R = out["R"][0].cpu().numpy()
            corners = out["corners_uv"][0].cpu().numpy() * [W0, H0]

            # --- ALL gates from the seg head ---------------------------------
            seg = torch.sigmoid(out["seg_logit"])[0, 0].cpu().numpy()
            seg_full = cv2.resize(seg, (W0, H0),
                                  interpolation=cv2.INTER_LINEAR)
            binm = (seg_full > 0.5).astype(np.uint8)
            n_cc, cc = cv2.connectedComponents(binm)
            dets, quads = [], []
            for ci in range(1, n_cc):
                comp = (cc == ci).astype(np.uint8)
                if comp.sum() < 120:
                    continue
                q = quad_from_mask(comp * 255)
                if q is None:
                    continue
                pos = pnp_pos(q, K)
                if pos is None:
                    continue
                dets.append(pos)
                quads.append((q, comp))
            assigned = tracks.update(dets)

            # --- left panel ---------------------------------------------------
            disp = cv2.resize(img, (VW, VH))
            sx, sy = VW / W0, VH / H0
            overlay = disp.copy()
            for i, (q, comp) in enumerate(quads):
                tid = assigned.get(i, -1)
                col = TRACK_COLORS[tid % len(TRACK_COLORS)]
                comp_d = cv2.resize(comp, (VW, VH),
                                    interpolation=cv2.INTER_NEAREST)
                overlay[comp_d > 0] = col
                for (u, v) in q:
                    cv2.circle(disp, (int(u * sx), int(v * sy)), 5, col, -1)
                    cv2.circle(disp, (int(u * sx), int(v * sy)), 5,
                               (0, 0, 0), 1)
                c0 = q.mean(axis=0)
                cv2.putText(disp, f"G{tid} {dets[i][2]:.1f}m",
                            (int(c0[0] * sx), int(c0[1] * sy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1,
                            cv2.LINE_AA)
            disp = cv2.addWeighted(overlay, 0.35, disp, 0.65, 0)

            # target head: EMA-smoothed keypoint markers (no raw quad)
            chk = filter_gate_estimate(corners, t_pos, K, R_direct=R)
            if vis >= 0.5:
                ema_kp = corners if ema_kp is None else \
                    0.6 * ema_kp + 0.4 * corners
                for (u, v) in ema_kp:
                    cv2.drawMarker(disp, (int(u * sx), int(v * sy)),
                                   (0, 255, 255), cv2.MARKER_TILTED_CROSS,
                                   11, 2)
                if chk.consistent and chk.corners_snapped is not None:
                    sp = (chk.corners_snapped * [sx, sy]).astype(int)
                    cv2.polylines(disp, [sp.reshape(-1, 1, 2)], True,
                                  (0, 255, 255), 1)
            else:
                ema_kp = None

            jump = (float(np.linalg.norm(t_pos - prev_tpos))
                    if prev_tpos is not None else 0.0)
            prev_tpos = t_pos
            telem = [
                f"t={fi}  gates tracked: "
                f"{sum(1 for tr in tracks.tracks.values() if tr['age']==0)}",
                f"target d={np.linalg.norm(t_pos):5.2f} m  vis={vis:.2f}",
                f"pnp resid={chk.reproj_residual_px:6.1f} px  "
                f"{'OK' if chk.consistent else '--'}"
                if np.isfinite(chk.reproj_residual_px) else "pnp resid n/a",
                f"agree={chk.agreement_m:5.2f} m"
                if np.isfinite(chk.agreement_m) else "agree n/a",
                f"jump={jump:5.2f} m/frame",
            ]
            mpanel = draw_map(VH, MPW, tracks,
                              t_pos if vis >= 0.25 else None, vis, telem)
            vw.write(np.hstack([disp, mpanel]))
    vw.release()
    print(f"-> {out_dir}/track_map.mp4  ({len(paths)} frames)")


if __name__ == "__main__":
    main()
