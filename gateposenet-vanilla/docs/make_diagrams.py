#!/usr/bin/env python3
"""Render the GatePoseNet-MG documentation diagrams.

Produces two polished figures used by gateposenet/README.md:

    docs/img/gateposenet_mg_arch.png   block architecture with tensor shapes
    docs/img/gateposenet_mg_io.png     output geometry (3-D) + output routing

Reproducible and dependency-light (matplotlib + numpy). Run:

    uv run python docs/make_diagrams.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "img")
os.makedirs(IMG, exist_ok=True)

# ---- shared style -------------------------------------------------------- #
INK, MUTED, BG = "#2A2A32", "#6B6B76", "#FCFCFB"
FAM = {
    "io":    ("#EEEEF0", "#9AA0AA"),   # neutral inputs / outputs
    "enc":   ("#DCE6F4", "#4C74A8"),   # CNN encoder
    "temp":  ("#F6E7CE", "#C28C3E"),   # temporal (ego / ConvGRU)
    "mask":  ("#D3E9E5", "#3C8B82"),   # pixel decoder / mask path
    "attn":  ("#DBEBD8", "#548A55"),   # tokens / transformer
    "head":  ("#E7DFF1", "#7C5CA6"),   # readout heads
}
for f in ("Helvetica", "Arial", "DejaVu Sans"):
    if any(f.lower() in n.lower() for n in
           (fp.name for fp in fm.fontManager.ttflist)):
        plt.rcParams["font.family"] = f
        break
plt.rcParams.update({"font.size": 9, "text.color": INK,
                     "axes.edgecolor": INK})


class Box:
    __slots__ = ("cx", "cy", "w", "h")

    def __init__(self, cx, cy, w, h):
        self.cx, self.cy, self.w, self.h = cx, cy, w, h

    def edge(self, side):
        x, y, w, h = self.cx, self.cy, self.w / 2, self.h / 2
        return {"l": (x - w, y), "r": (x + w, y),
                "t": (x, y + h), "b": (x, y - h)}[side]


def box(ax, cx, cy, w, h, title, detail="", fam="io", tsize=10.5,
        dsize=8.0, align="center"):
    fill, edge = FAM[fam]
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.10",
        linewidth=1.4, edgecolor=edge, facecolor=fill,
        mutation_aspect=1.0, zorder=2))
    if detail:
        ax.text(cx, cy + h * 0.30, title, ha="center", va="center",
                fontsize=tsize, fontweight="bold", zorder=3)
        ax.text(cx, cy - h * 0.14, detail, ha="center", va="center",
                fontsize=dsize, color="#3a3a42", linespacing=1.35, zorder=3)
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                fontsize=tsize, fontweight="bold", zorder=3,
                linespacing=1.35)
    return Box(cx, cy, w, h)


def arrow(ax, p0, p1, color=INK, lw=1.5, ls="-", rad=0.0, z=1.8):
    cs = "arc3,rad=%s" % rad
    ax.add_patch(FancyArrowPatch(
        p0, p1, connectionstyle=cs, arrowstyle="-|>", mutation_scale=13,
        linewidth=lw, color=color, linestyle=ls, shrinkA=0, shrinkB=0,
        zorder=z))


def elbow(ax, pts, color=INK, lw=1.5, z=1.8):
    """Orthogonal polyline with an arrowhead on the final segment."""
    for a, b in zip(pts[:-1], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw,
                solid_capstyle="round", zorder=z)
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, z=z)


def label(ax, x, y, s, size=8.0, style="italic", color=None, ha="center",
          rot=0):
    ax.text(x, y, s, fontsize=size, style=style, ha=ha, va="center",
            color=color or MUTED, zorder=4, rotation=rot)


# ========================================================================= #
#  Figure 1 — block architecture                                            #
# ========================================================================= #
def build_arch():
    fig, ax = plt.subplots(figsize=(17.5, 8.6), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 17.5)
    ax.set_ylim(0, 8.6)
    ax.axis("off")

    ax.text(8.75, 8.25, "GatePoseNet-MG", ha="center", fontsize=16,
            fontweight="bold")
    ax.text(8.75, 7.88, "temporal multi-gate decoder  ·  1.93 M params  ·  "
            "4.4 ms / frame TensorRT fp16  ·  step(image, ego, h) → "
            "outputs, h", ha="center", fontsize=9.5, color=MUTED)

    # -- inputs ------------------------------------------------------------
    rgb = box(ax, 1.35, 6.7, 2.05, 1.0, "RGB frame",
              "640×360 → 320×192\n/255, no mean/std", "io")
    ego = box(ax, 1.35, 1.35, 2.05, 1.1, "ego vector (7)",
              "[v 3 m/s, ω 3 rad/s, dt s]\nfrom flight-controller EKF",
              "io")

    # -- CNN encoder (vertical pyramid) -----------------------------------
    label(ax, 4.05, 7.55, "GateNet CNN encoder", size=9.5, style="italic",
          color="#4C74A8")
    ys = [6.7, 5.85, 5.0, 4.15, 3.3]
    txt = [("inc", "3 → 8 ch    320×192"),
           ("down1", "8 → 16    160×96"),
           ("down2", "16 → 32    80×48   · skip"),
           ("down3", "32 → 64    40×24   · skip"),
           ("down4", "64 → 128   20×12  bottleneck")]
    enc = []
    for y, (t, d) in zip(ys, txt):
        enc.append(box(ax, 4.05, y, 2.35, 0.74, t, d, "enc",
                       tsize=9.5, dsize=7.6))
    arrow(ax, rgb.edge("r"), enc[0].edge("l"))
    for a, b in zip(enc[:-1], enc[1:]):
        arrow(ax, a.edge("b"), b.edge("t"))

    # -- ego MLP -----------------------------------------------------------
    emlp = box(ax, 4.05, 1.35, 2.35, 0.95, "ego MLP",
               "7 → 64 → 128\nbroadcast add", "temp",
               tsize=9.5, dsize=7.8)
    arrow(ax, ego.edge("r"), emlp.edge("l"))

    # -- ConvGRU (temporal memory) ----------------------------------------
    gru = box(ax, 7.35, 3.3, 2.45, 1.35, "ConvGRU cell",
              "z,r,ĥ : 3×3 conv\nstate (128, 12, 20)\n"
              "TEMPORAL MEMORY", "temp", tsize=10.5, dsize=8.0)
    arrow(ax, enc[-1].edge("r"), gru.edge("l"))
    label(ax, 6.1, 3.62, "features", size=7.6)
    # ego -> ConvGRU left face, routed up a clear channel
    elbow(ax, [emlp.edge("r"), (5.75, 1.35), (5.75, 3.05),
               (gru.cx - gru.w / 2, 3.05)], color="#C28C3E")
    label(ax, 5.75, 2.35, "+ per-channel\nbias", size=7.4, color="#9a7233")
    # recurrent self-loop
    lb = gru.edge("b")
    ax.add_patch(FancyArrowPatch(
        (lb[0] + 0.85, lb[1]), (lb[0] - 0.85, lb[1]),
        connectionstyle="arc3,rad=-0.75", arrowstyle="-|>",
        mutation_scale=12, linewidth=1.4, color="#C28C3E", zorder=1.6))
    label(ax, 7.35, 1.98, "hidden state h (128,12,20) — recycled each "
          "frame  (ONNX h_in / h_out)", size=7.6, color="#9a7233")

    # -- mask path (up the same column) -----------------------------------
    pdec = box(ax, 7.35, 5.35, 2.45, 0.98, "pixel decoder",
               "Up + skip ×2\n→ 1×1 conv → d=128", "mask",
               tsize=10.0, dsize=7.9)
    arrow(ax, gru.edge("t"), pdec.edge("b"))
    label(ax, 7.62, 4.72, "h(t)", size=7.6)
    arrow(ax, enc[2].edge("r"), pdec.edge("l"))          # down2 skip
    label(ax, 6.1, 5.62, "skips", size=7.6)
    mfeat = box(ax, 7.35, 7.15, 2.45, 0.95, "mask features",
                "(128, 48, 80)\n¼ resolution", "mask",
                tsize=10.0, dsize=7.9)
    arrow(ax, pdec.edge("t"), mfeat.edge("b"))

    # -- token / transformer path -----------------------------------------
    tok = box(ax, 10.15, 3.3, 2.4, 1.15, "tokenize",
              "12×20 → 240 tokens\n+ 2-D sine PE → d=128",
              "attn", tsize=10.0, dsize=7.9)
    arrow(ax, gru.edge("r"), tok.edge("l"))
    label(ax, 9.0, 3.6, "h(t)", size=7.6)
    qry = box(ax, 10.15, 5.65, 2.4, 0.9, "8 gate queries",
              "learned  (8, 128)", "attn", tsize=10.0, dsize=7.9)
    trf = box(ax, 12.75, 4.35, 2.45, 1.75, "transformer decoder ×2",
              "self-attn (queries)\ncross-attn → 240 tokens\n"
              "FFN 128 → 512 → 128\n4 heads, pre-norm", "attn",
              tsize=10.0, dsize=7.8)
    arrow(ax, tok.edge("r"), (trf.cx - trf.w / 2, 4.05))
    label(ax, 11.75, 3.75, "K,V", size=7.6)
    arrow(ax, qry.edge("r"), (trf.cx - trf.w / 2, 4.95))
    label(ax, 11.75, 5.45, "Q", size=7.6)

    # -- consolidated per-query readout -----------------------------------
    head = box(ax, 15.7, 4.05, 2.9, 4.55,
               "per-query readout  (×8)", "", "head", tsize=10.5)
    ax.text(head.cx, 5.55,
            "MLP  128 → 27\n"
            "presence·1   target·1\n"
            "corners·8   inside·4\n"
            "center·2   position·3\n"
            "log_depth·1  rot6d·6\n"
            "vis_frac·1", ha="center", va="center", fontsize=8.0,
            linespacing=1.4, zorder=3)
    ax.plot([head.cx - 1.3, head.cx + 1.3], [3.62, 3.62], color="#7C5CA6",
            lw=0.9, ls=(0, (4, 3)), zorder=3)
    ax.text(head.cx, 2.98,
            "instance mask\n"
            "mask-embed(q) ⊗ mask features\n→ (8, 48, 80) logits",
            ha="center", va="center", fontsize=8.0, linespacing=1.4,
            zorder=3)
    arrow(ax, trf.edge("r"), (head.cx - head.w / 2, 4.35))
    # mask features skip, routed cleanly along the top edge
    elbow(ax, [mfeat.edge("r"), (14.05, 7.15), (14.05, 6.35),
               (head.cx, 6.35)], color="#3C8B82")
    label(ax, 11.4, 7.4, "mask features (¼-res) → instance masks",
          size=7.8, color="#2f6f68", ha="center")

    # -- outputs -----------------------------------------------------------
    out = box(ax, 10.6, 0.55, 10.5, 0.86,
              "OUTPUTS per query  ·  sigmoid(presence)≥0.5 = gate "
              "·  softmax(target) = gate flown at  ·  corners_uv / "
              "center_uv ×[W,H]→px  ·  position [m] camera "
              "frame  ·  R = GS(rot6d), fly-through = R[:,2]  ·  "
              "sigmoid(mask)>0.5  ·  visible_frac ∈ [0,1]", "",
              "io", tsize=8.2)
    arrow(ax, head.edge("b"), (head.cx, out.cy + out.h / 2))

    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(IMG, "gateposenet_mg_arch.png"),
                facecolor=BG, bbox_inches="tight")
    plt.close(fig)


# ========================================================================= #
#  Figure 2 — output geometry (3-D) + routing                               #
# ========================================================================= #
def _R(yaw, pitch):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return Ry @ Rx


def build_io():
    fig, (axg, axr) = plt.subplots(
        1, 2, figsize=(17.5, 8.2), dpi=150,
        gridspec_kw={"width_ratios": [1.05, 1.0]})
    fig.patch.set_facecolor(BG)

    # ---------------- left: 3-D output geometry -----------------------
    ax = axg
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_title("Output geometry — one gate query",
                 fontsize=13, fontweight="bold", pad=12)

    def P(p):                       # camera-optical -> screen
        x, y, z = p
        return np.array([x + 0.42 * z, -y + 0.60 * z])

    O = np.array([0.0, 0.0, 0.0])
    # camera frustum + image plane
    d = 2.15
    corners = [(-1.02, -0.62, d), (1.02, -0.62, d),
               (1.02, 0.62, d), (-1.02, 0.62, d)]
    for c in corners:
        ax.plot(*zip(P(O), P(c)), color="#C4C4CC", lw=0.9, zorder=1)
    poly = [P(c) for c in corners] + [P(corners[0])]
    ax.plot(*zip(*poly), color="#B4B4BE", lw=1.1, zorder=1)
    label(ax, *(P((-1.02, 0.62, d)) + [-0.15, -0.25]),
          "image plane\ncorners_uv, center_uv\n(normalized ×[W,H])",
          size=7.6, ha="right")

    # camera-optical axes
    A = 1.35
    for vec, col, name, off in [
            ((1, 0, 0), "#C0392B", "+X right", (0.18, -0.05)),
            ((0, 1, 0), "#1E8449", "+Y down", (0.05, -0.22)),
            ((0, 0, 1), "#1F8Fb0", "+Z forward", (0.10, 0.16))]:
        tip = P((A * vec[0], A * vec[1], A * vec[2]))
        ax.add_patch(FancyArrowPatch(P(O), tip, arrowstyle="-|>",
                     mutation_scale=15, lw=2.2, color=col, zorder=5))
        ax.text(tip[0] + off[0], tip[1] + off[1], name, color=col,
                fontsize=8.6, fontweight="bold",
                ha="left" if off[0] >= 0 else "right", va="center", zorder=6)

    # gate
    t = np.array([0.95, -0.85, 3.55])
    R = _R(np.deg2rad(24), np.deg2rad(-11))

    def quad(half):
        loc = np.array([[-half, -half, 0], [half, -half, 0],
                        [half, half, 0], [-half, half, 0]])
        return [P(R @ p + t) for p in loc]

    for half, col, lw, fill in [(1.35, "#E8541C", 3.4, "#F2C9B4"),
                                (0.75, "#E8781C", 2.2, None)]:
        q = quad(half)
        if fill:
            ax.add_patch(plt.Polygon(q, closed=True, facecolor=fill,
                         edgecolor="none", alpha=0.35, zorder=2))
        ax.add_patch(plt.Polygon(q, closed=True, fill=False,
                     edgecolor=col, lw=lw, zorder=4,
                     joinstyle="round"))
        for pt in q:
            ax.plot(*pt, "o", ms=5, mfc="white", mec=col, mew=1.6, zorder=5)

    cen = P(t)
    ax.plot(*cen, "+", ms=13, mew=2.2, color=INK, zorder=6)
    # gate-frame axes from centre
    for vec, col, name, ln in [(R[:, 0], "#C0392B", "R[:,0]", 0.95),
                               (R[:, 1], "#1E8449", "R[:,1]", 0.95),
                               (R[:, 2], "#7C3FB0", "fly-through R[:,2]",
                                1.5)]:
        tip = P(t + vec * ln)
        ax.add_patch(FancyArrowPatch(cen, tip, arrowstyle="-|>",
                     mutation_scale=13,
                     lw=2.4 if "fly" in name else 1.7, color=col, zorder=5))
        dx = 0.16 if "fly" in name else 0.10
        ax.annotate(name, tip, xytext=(tip[0] + dx, tip[1] + 0.22),
                    fontsize=8.0 if "fly" not in name else 8.6,
                    color=col, ha="left",
                    fontweight="bold" if "fly" in name else "normal",
                    zorder=6)

    # position vector
    ax.add_patch(FancyArrowPatch(P(O), cen, arrowstyle="-|>",
                 mutation_scale=13, lw=1.6, color=INK, ls=(0, (5, 3)),
                 zorder=4))
    mid = 0.5 * (P(O) + cen)
    ax.annotate("position (x,y,z) [m] → gate centre\n‖position‖ = range",
                mid, xytext=(mid[0] - 1.75, mid[1] + 0.35), fontsize=8.2,
                ha="center", color=INK,
                arrowprops=dict(arrowstyle="-", color="#9AA0AA", lw=0.9))

    # corner callouts with leaders
    q_out = quad(1.35)
    ax.annotate("outer corners — 2.7 m square", q_out[0],
                xytext=(q_out[0][0] - 0.2, q_out[0][1] + 0.9), fontsize=8.2,
                ha="center", color="#8a3210",
                arrowprops=dict(arrowstyle="-", color="#c0714f", lw=0.9))
    q_in = quad(0.75)
    ax.annotate("inner corners — 1.5 m square", q_in[2],
                xytext=(q_in[2][0] + 1.35, q_in[2][1] - 0.15), fontsize=8.2,
                ha="center", color="#8a5210",
                arrowprops=dict(arrowstyle="-", color="#c08a4f", lw=0.9))
    # frame + ego caption, pinned to a clear corner
    ax.text(0.015, 0.02,
            "CAMERA OPTICAL FRAME  —  every 3-D output lives here\n"
            "+X right, +Y down, +Z forward (m).  R = R_cam_gate, defined "
            "modulo the ring's D4 symmetry.\n"
            "ego input [v_cam, ω_cam, dt] drives the memory across "
            "blind frames.",
            transform=ax.transAxes, fontsize=8.2, va="bottom", ha="left",
            color="#33333b",
            bbox=dict(boxstyle="round,pad=0.5", fc="#F3F3F1",
                      ec="#C8C8D0", lw=1.0))
    ax.autoscale_view()
    ax.margins(0.14)

    # ---------------- right: routing ----------------------------------
    ax = axr
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Where each output goes", fontsize=13, fontweight="bold",
                 pad=12)

    src = box(ax, 1.7, 5.0, 2.5, 1.35, "GatePoseNet-MG",
              "step(image, ego, h)\n4.4 ms  TRT fp16", "attn",
              tsize=10.5, dsize=8.2)
    rows = [
        (8.55, "mask (per gate)", "sigmoid>0.5 · 48×80 → up",
         "mask", "viz / VIO / gsplat", "map inputs"),
        (6.9, "corners_uv ×4 + inside", "center_uv ×[W,H]→px",
         "head", "geometric_filter", "PnP check / correct\nresidual + "
         "agreement"),
        (5.0, "position [m], R (3×3)", "log_depth [ln m]", "head",
         "body_frame.py", "R(tilt) → BODY-NED\n→ controller"),
        (3.1, "presence (sigmoid)", "target (softmax / argmax)", "head",
         "gate selection", "which gate to fly"),
        (1.45, "visible_frac", "∈ [0, 1]", "head", "trust gating",
         "when to believe 2-D"),
    ]
    mids, dsts = [], []
    for y, t1, t2, fam, c1, c2 in rows:
        m = box(ax, 5.15, y, 2.75, 1.28, t1, t2, fam, tsize=9.0, dsize=7.8)
        dd = box(ax, 8.55, y, 2.6, 1.28, c1, c2, "io", tsize=9.2,
                 dsize=7.6)
        mids.append(m)
        dsts.append(dd)
        arrow(ax, m.edge("r"), dd.edge("l"))
    for m in mids:
        r = src.edge("r")
        arrow(ax, r, (m.cx - m.w / 2, m.cy), rad=0.0)
    # geometric_filter cross-checks body_frame position
    arrow(ax, dsts[1].edge("b"), dsts[2].edge("t"), color="#7C5CA6",
          lw=1.3)
    label(ax, 9.55, 5.95, "cross-checks\nposition", size=7.4,
          color="#6b4f92", ha="left")

    fig.tight_layout(pad=0.6, w_pad=1.5)
    fig.savefig(os.path.join(IMG, "gateposenet_mg_io.png"),
                facecolor=BG, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    build_arch()
    build_io()
    print("wrote", os.path.join(IMG, "gateposenet_mg_arch.png"))
    print("wrote", os.path.join(IMG, "gateposenet_mg_io.png"))
