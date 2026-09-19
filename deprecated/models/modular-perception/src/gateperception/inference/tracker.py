from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment

from .decoder import Detection


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den > 1e-8 else 0.0


@dataclass
class TrackState:
    track_id: str
    embedding: np.ndarray
    mask: np.ndarray
    score: float
    last_frame: int
    age: int = 0


class GateTracker:
    def __init__(self, max_age: int = 8, match_threshold: float = 0.75, embedding_weight: float = 0.65, mask_iou_weight: float = 0.35, ema: float = 0.8):
        self.max_age = int(max_age)
        self.match_threshold = float(match_threshold)
        self.embedding_weight = float(embedding_weight)
        self.mask_iou_weight = float(mask_iou_weight)
        self.ema = float(ema)
        self.tracks: dict[str, TrackState] = {}
        self._next = 1

    def _new_id(self) -> str:
        tid = f"track_{self._next:04d}"
        self._next += 1
        return tid

    def update(self, frame_index: int, detections: list[Detection]) -> list[Detection]:
        active_ids = [tid for tid, tr in self.tracks.items() if frame_index - tr.last_frame <= self.max_age]
        active = [self.tracks[tid] for tid in active_ids]
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        if active and detections:
            cost = np.zeros((len(active), len(detections)), dtype=np.float32)
            for i, tr in enumerate(active):
                for j, det in enumerate(detections):
                    sim = cosine_similarity(tr.embedding, det.track_embedding)
                    iou = mask_iou(tr.mask, det.mask)
                    cost[i, j] = self.embedding_weight * (1.0 - sim) + self.mask_iou_weight * (1.0 - iou)
            rr, cc = linear_sum_assignment(cost)
            for i, j in zip(rr, cc):
                if float(cost[i, j]) > self.match_threshold:
                    continue
                tr = active[i]
                det = detections[j]
                det.track_id = tr.track_id
                emb = self.ema * tr.embedding + (1.0 - self.ema) * det.track_embedding
                norm = np.linalg.norm(emb)
                if norm > 1e-8:
                    emb /= norm
                tr.embedding = emb
                tr.mask = det.mask
                tr.score = det.score
                tr.last_frame = frame_index
                tr.age = 0
                matched_tracks.add(i)
                matched_dets.add(j)
        for j, det in enumerate(detections):
            if j in matched_dets:
                continue
            tid = self._new_id()
            det.track_id = tid
            self.tracks[tid] = TrackState(tid, det.track_embedding.copy(), det.mask.copy(), det.score, frame_index)
        dead = []
        for tid, tr in self.tracks.items():
            if tr.last_frame != frame_index:
                tr.age = frame_index - tr.last_frame
            if tr.age > self.max_age:
                dead.append(tid)
        for tid in dead:
            del self.tracks[tid]
        return detections

    def memory_only(self, frame_index: int, max_emit_age: int = 2, confidence_decay: float = 0.72) -> list[Detection]:
        out: list[Detection] = []
        for tr in self.tracks.values():
            age = frame_index - tr.last_frame
            if age <= 0 or age > max_emit_age:
                continue
            ys, xs = np.nonzero(tr.mask)
            if len(xs) == 0:
                continue
            box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
            # No visible segmentation is claimed on a memory-only frame. The old box is used only
            # to extract the current RGB crop for temporal keypoint inference.
            empty = np.zeros_like(tr.mask, dtype=bool)
            out.append(Detection(
                mask_id=None,
                score=float(tr.score * (confidence_decay ** age)),
                mask=empty,
                box_xyxy_px=box,
                track_embedding=tr.embedding.copy(),
                track_id=tr.track_id,
                source="memory_propagated",
            ))
        return out
