"""QuAdGate: sub-pixel gate-corner extraction from segmentation masks.

Reproduces the algorithm from MonoRace (Materials & Methods > QuAdGate):

  1. (optional) derotate the mask by the estimated roll so the image vertical
     axis matches the world up-direction.
  2. detect line segments with the OpenCV Line Segment Detector (LSD), params
     scale=0.8, sigma_scale=0.8, quant=25.0, ang_th=30.0.
  3. extend each segment so its total length grows by a factor 5/3.
  4. intersect extended (near-horizontal x near-vertical) lines -> corner
     candidates. Using line intersections (not mask extremities) makes the
     corner location robust to rounded / damaged masks.
  5. build a 4-value descriptor [v_TL, v_TR, v_BR, v_BL] by sampling the
     thresholded mask 5 px along the two line directions.
  6. match candidates to priors (projected from the state estimate): require an
     exact descriptor (corner-type) match, candidate within 100 px of the prior,
     then a RANSAC partial-affine fit (4 DoF, RansacThreshold=5.0); reject if
     the recovered translation exceeds 150 px.

If LSD is unavailable in the installed OpenCV build, a probabilistic Hough
fallback is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# LSD parameters from the paper
_LSD_SCALE = 0.8
_LSD_SIGMA_SCALE = 0.8
_LSD_QUANT = 25.0
_LSD_ANG_TH = 30.0

# matching parameters from the paper
EXTEND_FACTOR = 5.0 / 3.0
DESC_OFFSET_PX = 5
MAX_PRIOR_DIST_PX = 100.0
RANSAC_THRESH = 5.0
MAX_TRANSLATION_PX = 150.0


@dataclass
class GateCornerPrior:
    """A projected prior gate corner used for matching."""
    xy: np.ndarray                 # (2,) predicted pixel location
    descriptor: tuple              # (vTL, vTR, vBR, vBL) in {0,1}
    gate_id: int = 0
    corner_id: int = 0             # 0..7 (4 inner + 4 outer, say)


@dataclass
class MatchedCorner:
    xy: np.ndarray
    prior: GateCornerPrior
    descriptor: tuple


def _make_lsd():
    try:
        return cv2.createLineSegmentDetector(
            cv2.LSD_REFINE_STD, _LSD_SCALE, _LSD_SIGMA_SCALE,
            _LSD_QUANT, _LSD_ANG_TH,
        )
    except Exception:
        return None


def detect_lines(mask: np.ndarray) -> np.ndarray:
    """Return Nx4 array of (x1,y1,x2,y2) line segments from a binary mask."""
    m = mask
    if m.dtype != np.uint8:
        m = (m > 0.5).astype(np.uint8) * 255
    elif m.max() <= 1:
        m = m * 255
    lsd = _make_lsd()
    if lsd is not None:
        lines, _, _, _ = lsd.detect(m)
        if lines is None:
            return np.zeros((0, 4), np.float32)
        return lines.reshape(-1, 4).astype(np.float32)
    # Hough fallback
    edges = cv2.Canny(m, 50, 150)
    hl = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                         minLineLength=20, maxLineGap=8)
    if hl is None:
        return np.zeros((0, 4), np.float32)
    return hl.reshape(-1, 4).astype(np.float32)


def _extend(seg, factor=EXTEND_FACTOR):
    x1, y1, x2, y2 = seg
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    return np.array([cx - dx / 2 * factor, cy - dy / 2 * factor,
                     cx + dx / 2 * factor, cy + dy / 2 * factor], np.float32)


def _angle(seg):
    return np.degrees(np.arctan2(seg[3] - seg[1], seg[2] - seg[0]))


def _intersect(a, b):
    """Intersection of two line segments treated as infinite lines."""
    x1, y1, x2, y2 = a
    x3, y3, x4, y4 = b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return np.array([px, py], np.float32)


def _unit(seg):
    d = np.array([seg[2] - seg[0], seg[3] - seg[1]], np.float32)
    n = np.linalg.norm(d)
    return d / n if n > 1e-6 else d


class QuAdGate:
    def __init__(self, angle_tol_deg: float = 45.0, threshold: float = 0.5):
        self.angle_tol = angle_tol_deg
        self.threshold = threshold

    # -- descriptor --------------------------------------------------------
    def _descriptor(self, mask_bin, corner, h_dir, v_dir):
        # h_dir points "right", v_dir points "down" in (derotated) image
        h = h_dir * DESC_OFFSET_PX
        v = v_dir * DESC_OFFSET_PX
        pts = {
            "TL": corner - h - v,
            "TR": corner + h - v,
            "BR": corner + h + v,
            "BL": corner - h + v,
        }
        H, W = mask_bin.shape
        vals = []
        for key in ("TL", "TR", "BR", "BL"):
            x, y = pts[key]
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < W and 0 <= yi < H:
                vals.append(int(mask_bin[yi, xi] > 0))
            else:
                vals.append(0)
        return tuple(vals)

    # -- candidate generation ---------------------------------------------
    def candidates(self, mask: np.ndarray):
        """Return list of dicts: {xy, descriptor}."""
        mask_bin = (mask > self.threshold).astype(np.uint8) if mask.dtype != np.uint8 \
            else (mask > 0).astype(np.uint8)
        H, W = mask_bin.shape
        lines = detect_lines(mask)
        if len(lines) == 0:
            return []

        horiz, vert = [], []
        for seg in lines:
            ang = abs(_angle(seg)) % 180
            ang = min(ang, 180 - ang)  # fold to [0,90]
            if ang <= self.angle_tol:
                horiz.append(_extend(seg))
            else:
                vert.append(_extend(seg))

        out = []
        for hseg in horiz:
            hdir = _unit(hseg)
            if hdir[0] < 0:
                hdir = -hdir  # point right
            for vseg in vert:
                pt = _intersect(hseg, vseg)
                if pt is None:
                    continue
                if not (-5 <= pt[0] < W + 5 and -5 <= pt[1] < H + 5):
                    continue
                vdir = _unit(vseg)
                if vdir[1] < 0:
                    vdir = -vdir  # point down
                desc = self._descriptor(mask_bin, pt, hdir, vdir)
                out.append({"xy": pt, "descriptor": desc})
        return out

    # -- matching ----------------------------------------------------------
    def match(self, candidates, priors):
        """Match candidates to priors and refine with RANSAC partial-affine.

        Returns list[MatchedCorner].
        """
        if not candidates or not priors:
            return []

        # 1) coarse matches: same descriptor (corner type), within 100 px
        prior_pts, cand_pts, pairs = [], [], []
        for pr in priors:
            best, best_d = None, MAX_PRIOR_DIST_PX
            for c in candidates:
                if c["descriptor"] != tuple(pr.descriptor):
                    continue
                d = float(np.linalg.norm(c["xy"] - pr.xy))
                if d < best_d:
                    best, best_d = c, d
            if best is not None:
                prior_pts.append(pr.xy)
                cand_pts.append(best["xy"])
                pairs.append((pr, best))

        if len(pairs) < 2:
            return [MatchedCorner(c["xy"], pr, c["descriptor"]) for pr, c in pairs]

        prior_pts = np.asarray(prior_pts, np.float32)
        cand_pts = np.asarray(cand_pts, np.float32)

        # 2) RANSAC partial-affine (translation, rotation, uniform scale)
        M, inliers = cv2.estimateAffinePartial2D(
            prior_pts, cand_pts, method=cv2.RANSAC,
            ransacReprojThreshold=RANSAC_THRESH,
        )
        if M is None:
            return []
        if abs(M[0, 2]) > MAX_TRANSLATION_PX or abs(M[1, 2]) > MAX_TRANSLATION_PX:
            return []  # poor convergence

        inliers = inliers.ravel().astype(bool) if inliers is not None \
            else np.ones(len(pairs), bool)
        results = []
        for keep, (pr, c) in zip(inliers, pairs):
            if keep:
                results.append(MatchedCorner(c["xy"], pr, c["descriptor"]))
        return results

    def __call__(self, mask, priors=None, roll_rad: float = 0.0):
        """Full pipeline. If ``roll_rad`` given, derotate the mask first and map
        results back to the original frame."""
        if abs(roll_rad) > 1e-6:
            H, W = mask.shape
            R = cv2.getRotationMatrix2D((W / 2, H / 2), np.degrees(roll_rad), 1.0)
            mask_d = cv2.warpAffine(mask.astype(np.float32), R, (W, H))
            Rinv = cv2.invertAffineTransform(R)
        else:
            mask_d, Rinv = mask, None

        cands = self.candidates(mask_d)
        if priors is None:
            if Rinv is not None:
                for c in cands:
                    c["xy"] = (Rinv[:, :2] @ c["xy"] + Rinv[:, 2]).astype(np.float32)
            return cands
        matched = self.match(cands, priors)
        if Rinv is not None:
            for m in matched:
                m.xy = (Rinv[:, :2] @ m.xy + Rinv[:, 2]).astype(np.float32)
        return matched
