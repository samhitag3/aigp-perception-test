"""PnP CHECK / CORRECT layer for the deep model's gate estimates.

Division of labor: the DEEP MODEL is the estimator — it outputs corners
(including inferring occluded/off-frame ones; it is explicitly trained for
that), metric position and rotation. This module does NOT re-estimate; it
uses the known rigid 2.7 m square to CHECK and, where cheap, CORRECT:

* **check** — fit the rigid-gate manifold (PnP) to the model's corners; the
  RMS reprojection **residual** says whether the predicted corner set is
  geometrically possible (high residual => don't trust the 2-D outputs this
  frame), and the PnP-vs-direct-head position **agreement** cross-checks the
  two independent output paths (they fail differently);
* **correct** — ``corners_snapped`` replaces the raw corners with the fitted
  pose's reprojection: exact side lengths / a correctly warped square by
  construction. Position stays the model's direct head unless the corner
  set is highly consistent AND agrees — then the two are averaged.

Measured on synthetic test (all visible frames): direct head 0.585 m median,
raw PnP 0.649 m, checked fusion 0.583 m — i.e. the check never hurts and the
residual/agreement signals are the real product.

Subset fallback: if the caller masks corners (e.g. by the model's own
per-corner confidence), the check still runs from 3 corners (P3P,
prior-disambiguated) or 2 (rotation-only refine around the model's R with
the model's position anchored) — still checking/correcting the model, never
replacing it.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

AIGP_GATE_OUTER_M = 2.7


def _gate_object_points(gate_m: float = AIGP_GATE_OUTER_M) -> np.ndarray:
    h = gate_m / 2.0
    # gate body frame: +X right, +Y down, +Z fly-through; TL,TR,BR,BL
    return np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]],
                    dtype=np.float64)


@dataclass
class GateEstimate:
    """Output of :func:`filter_gate_estimate`."""

    position_fused: np.ndarray        # (3,) best position estimate (m)
    position_direct: np.ndarray       # network metric head, passthrough
    position_pnp: np.ndarray | None   # from corners (None if PnP failed)
    corners_snapped: np.ndarray | None  # (4,2) on the rigid-gate manifold
    reproj_residual_px: float         # corner-set consistency (inf if failed)
    agreement_m: float                # |direct - pnp| (inf if failed)
    used_pnp: bool                    # fused estimate blended PnP in
    consistent: bool                  # residual below threshold


def _rodrigues(w: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(w, dtype=np.float64).reshape(3, 1))
    return R


def _project(obj: np.ndarray, R: np.ndarray, t: np.ndarray,
             K: np.ndarray) -> np.ndarray:
    Xc = obj @ R.T + t
    return (Xc @ K.T / Xc[:, 2:3])[:, :2]


def _refine_rotation_fixed_t(obj_sub, img_sub, R0, t, K,
                             iters: int = 20) -> np.ndarray:
    """Gauss-Newton on so(3): rotation-only reprojection refine with the
    translation anchored (2-corner case: 4 residuals, 3 params)."""
    w = cv2.Rodrigues(R0)[0].reshape(3)
    for _ in range(iters):
        R = _rodrigues(w)
        r0 = (_project(obj_sub, R, t, K) - img_sub).ravel()
        J = np.zeros((r0.size, 3))
        eps = 1e-5
        for j in range(3):
            wp = w.copy()
            wp[j] += eps
            rj = (_project(obj_sub, _rodrigues(wp), t, K) - img_sub).ravel()
            J[:, j] = (rj - r0) / eps
        try:
            dw = np.linalg.lstsq(J, -r0, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        w = w + dw
        if np.linalg.norm(dw) < 1e-8:
            break
    return _rodrigues(w)


def filter_gate_estimate(
    corners_px: np.ndarray,
    position_direct: np.ndarray,
    K: np.ndarray,
    corner_mask: np.ndarray | None = None,
    R_direct: np.ndarray | None = None,
    gate_m: float = AIGP_GATE_OUTER_M,
    residual_gate_px: float = 6.0,
    agreement_gate_m: float = 1.0,
) -> GateEstimate:
    """Enforce the known rigid-square geometry using WHATEVER corners exist.

    The gate is a rigid square, so its perspective image is fully determined
    by far fewer than 4 corners — the fit degrades gracefully with the number
    of TRUSTED corners (``corner_mask``, e.g. the model's per-corner
    inside/confidence head or occlusion knowledge):

    * 4 corners: IPPE PnP (full 6-DoF, closed form).
    * 3 corners: P3P (up to 4 solutions) disambiguated by the direct-head
      position prior; full 6-DoF.
    * 2 corners: translation anchored at the direct-head position, rotation
      refined on so(3) from the two corner rays (4 residuals / 3 params),
      seeded by ``R_direct`` (the network's rotation head) when given.
    * <2 corners: geometry can't help; passthrough.

    In every case the fitted pose reprojects ALL FOUR object corners —
    ``corners_snapped`` therefore COMPLETES occluded/off-frame corners from
    the visible subset (rigid or correctly-warped square by construction).
    The consistency ``residual`` is computed over the trusted subset only.

    Parameters
    ----------
    corners_px : (4, 2) predicted corners, pixels, TL,TR,BR,BL.
    position_direct : (3,) the network's direct metric position (camera frame).
    K : (3, 3) camera intrinsics (nominal is fine — ±4% barely moves this).
    corner_mask : (4,) bool — which corners to trust (default: all finite).
    R_direct : (3, 3) network rotation head, seeds the 2-corner refine.
    residual_gate_px : subsets with a best-fit rigid reprojection worse than
        this are declared inconsistent (occlusion/garbage corners).
    agreement_gate_m : PnP is only fused when it also agrees with the direct
        head within this distance.
    """
    p_dir = np.asarray(position_direct, dtype=np.float64).reshape(3)
    K = np.asarray(K, dtype=np.float64)
    obj = _gate_object_points(gate_m)
    c = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
    mask = (np.asarray(corner_mask, dtype=bool).reshape(4)
            if corner_mask is not None else np.isfinite(c).all(axis=1))
    mask = mask & np.isfinite(c).all(axis=1)
    n = int(mask.sum())
    obj_sub, img_sub = obj[mask], c[mask]

    R = t = None
    if n == 4:
        ok, rvec, tvec = cv2.solvePnP(obj_sub, img_sub, K, None,
                                      flags=cv2.SOLVEPNP_IPPE)
        if ok:
            R, t = _rodrigues(rvec.reshape(3)), tvec.reshape(3)
    elif n == 3:
        ok, rvecs, tvecs = cv2.solveP3P(obj_sub, img_sub, K, None,
                                        flags=cv2.SOLVEPNP_AP3P)
        if ok and len(tvecs):
            # up to 4 solutions: pick the one nearest the direct-head prior
            best = min(range(len(tvecs)),
                       key=lambda i: np.linalg.norm(tvecs[i].reshape(3)
                                                    - p_dir))
            R, t = (_rodrigues(rvecs[best].reshape(3)),
                    tvecs[best].reshape(3))
    elif n == 2:
        # anchor translation at the direct head; refine rotation from the
        # two corner rays, seeded by the network's rotation when available.
        R0 = (np.asarray(R_direct, dtype=np.float64).reshape(3, 3)
              if R_direct is not None else np.eye(3))
        t = p_dir.copy()
        R = _refine_rotation_fixed_t(obj_sub, img_sub, R0, t, K)

    if R is None or t is None or t[2] <= 0.05:
        return GateEstimate(p_dir, p_dir, None, None, float("inf"),
                            float("inf"), False, False)

    snapped = _project(obj, R, t, K)      # completes ALL 4 corners
    residual = float(np.sqrt(
        ((snapped[mask] - img_sub) ** 2).sum(axis=1).mean()))
    p_pnp = t
    agreement = float(np.linalg.norm(p_pnp - p_dir))
    consistent = residual < residual_gate_px

    # 2-corner mode borrowed the direct position, so it cannot "agree" as an
    # independent witness — only full/3-corner fits vote on position.
    if consistent and agreement < agreement_gate_m and n >= 3:
        fused = 0.5 * (p_dir + p_pnp)
        used = True
    else:
        fused = p_dir
        used = False
    return GateEstimate(fused, p_dir, p_pnp, snapped, residual, agreement,
                        used, consistent)
