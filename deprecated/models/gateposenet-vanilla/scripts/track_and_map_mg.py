#!/usr/bin/env python3
"""GatePoseNet-MG tracker: LEARNED per-gate masks + keypoints + map.

Everything on screen comes from the MODEL (no classical mask post-
processing): each active query paints its own instance mask and 4 keypoints;
PnP runs on the MODEL's per-gate keypoints (check/correct + map position);
the query the target head selects is highlighted. Right panel: top-down
camera-frame localization map with per-query-track trails + live telemetry.

    uv run python scripts/track_and_map_mg.py --video  <file.mp4> [--fps 30]
    uv run python scripts/track_and_map_mg.py --frames <dir>      [--fps 30]
    uv run python scripts/track_and_map_mg.py --dataset <synthetic dir>
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
from gateposenet.model import build_gateposenet_mg
from perception.geometric_filter import filter_gate_estimate
from scripts.track_and_map import Tracks, draw_map, TRACK_COLORS  # reuse

GATE_OUTER_H = 2.7 / 2.0     # outer corner half-extent (m)
GATE_INNER_H = 1.5 / 2.0     # inner opening half-extent (m)
# gate-frame keypoints: 4 OUTER + 4 INNER corners (TL,TR,BR,BL) + CENTER
_KP_GATE = np.array(
    [[-GATE_OUTER_H, -GATE_OUTER_H, 0], [GATE_OUTER_H, -GATE_OUTER_H, 0],
     [GATE_OUTER_H, GATE_OUTER_H, 0], [-GATE_OUTER_H, GATE_OUTER_H, 0],
     [-GATE_INNER_H, -GATE_INNER_H, 0], [GATE_INNER_H, -GATE_INNER_H, 0],
     [GATE_INNER_H, GATE_INNER_H, 0], [-GATE_INNER_H, GATE_INNER_H, 0],
     [0.0, 0.0, 0.0]], dtype=np.float64)


def project_gate_keypoints(R, t, K):
    """9 keypoints (outer x4, inner x4, center) projected through the
    MODEL's 6-DoF pose + the known gate geometry -> always a geometrically
    coherent (correctly warped) square pair. Returns (9,2) px or None."""
    Xc = _KP_GATE @ np.asarray(R, np.float64).T + np.asarray(t, np.float64)
    if np.any(Xc[:, 2] < 0.15):
        return None
    uv = (Xc @ np.asarray(K, np.float64).T)
    return uv[:, :2] / uv[:, 2:3]


def load_gt_stream(dataset_dir):
    """frame_file -> list of GT records (for the GT telemetry stream)."""
    import json
    gt_path = Path(dataset_dir) / "ground_truth.json"
    if not gt_path.is_file():
        return {}
    with open(gt_path) as f:
        gt = json.load(f)
    by_file: dict = {}
    for o in gt.get("objects", []):
        by_file.setdefault(o["file"], []).append(o)
    return by_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gateposenet_mg.yaml")
    ap.add_argument("--checkpoint", default="runs/gateposenet_mg/best.pt")
    ap.add_argument("--frames", default=None, help="directory of frame images")
    ap.add_argument("--video", default=None, help="video file (mp4/mov/...)")
    ap.add_argument("--dataset", default=None,
                    help="synthetic dataset dir (adds GT telemetry)")
    ap.add_argument("--out", default="runs/track_and_map_mg")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--fx", type=float, default=320.0)
    ap.add_argument("--presence-th", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    gt_stream = load_gt_stream(args.dataset) if args.dataset else {}

    # ---- frame source: a video file, or a directory of images ----------
    def _iter_video(path, limit):
        cap = cv2.VideoCapture(str(path))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok or (limit and i >= limit):
                break
            yield f"{i:06d}.png", frame
            i += 1
        cap.release()

    if args.video:
        cap0 = cv2.VideoCapture(str(args.video))
        n_frames = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok, first = cap0.read()
        cap0.release()
        if not ok:
            raise SystemExit(f"cannot read video: {args.video}")
        if args.max_frames:
            n_frames = min(n_frames, args.max_frames) if n_frames \
                else args.max_frames
        frame_iter = _iter_video(args.video, args.max_frames)
    else:
        frames_dir = Path(args.frames) if args.frames else \
            Path(args.dataset) / "images"
        paths = sorted(p for p in frames_dir.iterdir()
                       if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        if args.max_frames:
            paths = paths[:args.max_frames]
        if not paths:
            raise SystemExit(f"no frame images found in {frames_dir}")
        first = cv2.imread(str(paths[0]))
        n_frames = len(paths)
        frame_iter = ((p.name, cv2.imread(str(p))) for p in paths)

    cfg = load_config(args.config)
    device = pick_device(args.device)
    d = cfg["data"]
    mh, mw = int(d.get("height", 192)), int(d.get("width", 320))
    model = build_gateposenet_mg(cfg["model"]).to(device).eval()
    load_checkpoint(args.checkpoint, model, map_location=device)

    H0, W0 = first.shape[:2]
    VW, VH, MPW = 960, 540, 420
    K = np.array([[args.fx, 0, W0 / 2.0], [0, args.fx, H0 / 2.0],
                  [0, 0, 1.0]])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_dir / "track_map_mg.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                         (VW + MPW, VH))

    ego = torch.zeros(1, 7, device=device)
    ego[0, 6] = 1.0 / args.fps
    h = None
    tracks = Tracks()
    prev_tpos = None
    with torch.no_grad():
        for fi, (name, img) in enumerate(frame_iter):
            if img is None:
                continue
            rgb = cv2.cvtColor(cv2.resize(img, (mw, mh),
                                          interpolation=cv2.INTER_AREA),
                               cv2.COLOR_BGR2RGB)
            x = (torch.from_numpy(rgb.transpose(2, 0, 1))[None].float()
                 / 255.0).to(device)
            out, h = model.step(x, ego, h)

            prob = torch.sigmoid(out["presence_logit"])[0]     # (Q,)
            t_q = int(out["target_logit"][0].argmax())
            active = [q for q in range(prob.shape[0])
                      if float(prob[q]) >= args.presence_th]
            masks = torch.sigmoid(out["mask_logit"])[0]        # (Q,h4,w4)

            # per-gate PnP on the MODEL's keypoints -> map positions
            dets, det_q = [], []
            for q in active:
                cq = out["corners_uv"][0, q].cpu().numpy() * [W0, H0]
                pq = out["position"][0, q].cpu().numpy()
                Rq = out["R"][0, q].cpu().numpy()
                chk = filter_gate_estimate(cq, pq, K, R_direct=Rq)
                dets.append(chk.position_fused)
                det_q.append((q, cq, chk))
            assigned = tracks.update(dets)

            disp = cv2.resize(img, (VW, VH))
            sx, sy = VW / W0, VH / H0
            # MASK-FIRST rendering: one saturated color per instance, alpha
            # fill + bold contour, keypoints as haloed dots ON the mask.
            # Dedupe overlapping queries by mask IoU; require a real mask.
            # rank queries by presence; keep top-4; take each query's
            # LARGEST CONNECTED COMPONENT (kills fragmentation splatter);
            # dedupe by union-IoU against already-kept instances.
            order = sorted(range(len(det_q)),
                           key=lambda i: -float(prob[det_q[i][0]]))
            kept = []
            kept_masks = []
            for i in order[:6]:
                q, cq, chk = det_q[i]
                if float(prob[q]) < max(args.presence_th, 0.6):
                    continue
                mq = (masks[q].cpu().numpy() > 0.5).astype(np.uint8)
                if mq.sum() < 40:
                    continue
                n_cc, cc = cv2.connectedComponents(mq)
                if n_cc > 1:
                    sizes = [(cc == j).sum() for j in range(1, n_cc)]
                    mq = (cc == (1 + int(np.argmax(sizes)))).astype(bool)
                else:
                    mq = mq.astype(bool)
                if mq.sum() < 40:
                    continue
                dup = False
                for km in kept_masks:
                    iou = (mq & km).sum() / max(1, (mq | km).sum())
                    ctr_iou = (mq & km).sum() / max(1, min(mq.sum(),
                                                           km.sum()))
                    if iou > 0.3 or ctr_iou > 0.7:
                        dup = True
                        break
                if dup:
                    continue
                kept.append(i)
                kept_masks.append(mq)
                if len(kept) >= 4:
                    break
            overlay = disp.copy()
            for ki, i in enumerate(kept):
                q, cq, chk = det_q[i]
                tid = assigned.get(i, -1)
                col = TRACK_COLORS[tid % len(TRACK_COLORS)]
                mfull = cv2.resize(
                    kept_masks[ki].astype(np.uint8), (VW, VH),
                    interpolation=cv2.INTER_NEAREST).astype(bool)
                overlay[mfull] = col
                cnts, _ = cv2.findContours(mfull.astype(np.uint8),
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, cnts, -1, col, 3)
            disp = cv2.addWeighted(overlay, 0.45, disp, 0.55, 0)
            for ki, i in enumerate(kept):
                q, cq, chk = det_q[i]
                tid = assigned.get(i, -1)
                col = TRACK_COLORS[tid % len(TRACK_COLORS)]
                mfull = cv2.resize(
                    kept_masks[ki].astype(np.uint8), (VW, VH),
                    interpolation=cv2.INTER_NEAREST)
                cnts, _ = cv2.findContours(mfull, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(disp, cnts, -1, col, 2)
                # 8 PROJECTED corners (outer+inner) + center, from the
                # MODEL's pose + known 2.7/1.5 m geometry (always a coherent
                # warped square pair).
                Rq = out["R"][0, q].cpu().numpy()
                pq3 = out["position"][0, q].cpu().numpy()
                kps = project_gate_keypoints(Rq, pq3, K)
                if kps is not None:
                    kp = (kps * [sx, sy]).astype(int)
                    cv2.polylines(disp, [kp[:4].reshape(-1, 1, 2)], True,
                                  col, 1)
                    cv2.polylines(disp, [kp[4:8].reshape(-1, 1, 2)], True,
                                  col, 1)
                    for j in range(4):                # OUTER corners
                        cv2.circle(disp, tuple(kp[j]), 7, (255, 255, 255), -1)
                        cv2.circle(disp, tuple(kp[j]), 7, (0, 0, 0), 1)
                        cv2.circle(disp, tuple(kp[j]), 4, col, -1)
                    for j in range(4, 8):             # INNER corners
                        cv2.circle(disp, tuple(kp[j]), 5, (255, 255, 255), -1)
                        cv2.circle(disp, tuple(kp[j]), 3, col, -1)
                    cv2.drawMarker(disp, tuple(kp[8]), (255, 255, 255),
                                   cv2.MARKER_CROSS, 16, 3)  # CENTER
                    cv2.drawMarker(disp, tuple(kp[8]), col,
                                   cv2.MARKER_CROSS, 12, 1)
                ys, xs = np.nonzero(mfull)
                lx, ly = (int(xs.min()), max(18, int(ys.min()) - 8))                     if len(xs) else (int(cq[0][0] * sx), int(cq[0][1] * sy))
                lbl = f"G{tid}  {dets[i][2]:.1f} m"
                if q == t_q:
                    lbl = "TARGET  " + lbl
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX,
                                              0.55, 2)
                cv2.rectangle(disp, (lx - 2, ly - th - 4),
                              (lx + tw + 2, ly + 4), (0, 0, 0), -1)
                cv2.putText(disp, lbl, (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255) if q == t_q else col, 2,
                            cv2.LINE_AA)

            t_pos = out["position"][0, t_q].cpu().numpy()
            t_vis = float(out["visible_frac"][0, t_q])
            jump = (float(np.linalg.norm(t_pos - prev_tpos))
                    if prev_tpos is not None else 0.0)
            prev_tpos = t_pos
            telem = [
                f"t={fi}  gates: {len(kept)}/{len(active)} "
                f"(tracks {sum(1 for tr in tracks.tracks.values() if tr['age']==0)})",
                f"PRED target q{t_q}: d={np.linalg.norm(t_pos):5.2f} m  "
                f"vf={t_vis:.2f}",
                f"jump={jump:5.2f} m/frame",
            ]
            # GROUND-TRUTH telemetry stream (synthetic datasets)
            gt_objs = gt_stream.get(name, [])
            if gt_objs:
                g_t = next((o for o in gt_objs if o.get("is_target")),
                           gt_objs[0])
                gt_pos = np.asarray(g_t.get("position_cam") or [0, 0, 0])
                err = float(np.linalg.norm(t_pos - gt_pos))
                n_vis_gt = sum(1 for o in gt_objs if o.get("visible"))
                telem += [
                    "--- GROUND TRUTH ---",
                    f"GT gates: {len(gt_objs)} ({n_vis_gt} visible)",
                    f"GT target d={np.linalg.norm(gt_pos):5.2f} m  "
                    f"vf={g_t.get('visible_frac') or 0:.2f}",
                    f"POS ERR = {err:5.2f} m",
                ]
                # faint GT corner dots for every visible gate
                for o in gt_objs:
                    if o.get("visible") and o.get("corners"):
                        for (u, v) in o["corners"]:
                            cv2.circle(disp, (int(u * sx), int(v * sy)), 2,
                                       (255, 255, 255), -1)
            mpanel = draw_map(VH, MPW, tracks,
                              t_pos if len(active) else None, t_vis, telem)
            vw.write(np.hstack([disp, mpanel]))
    vw.release()
    print(f"-> {out_dir}/track_map_mg.mp4  ({n_frames} frames)")


if __name__ == "__main__":
    main()
