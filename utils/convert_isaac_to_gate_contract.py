#!/usr/bin/env python3
"""
Convert Isaac Sim drone-racing data into the canonical UAV gate-perception
dataset contract.

Expected input layout
---------------------
INPUT_ROOT/
  calibration.json
  dataset.json
  env_000/
    environment.json
    episode_0000000/
      control.jsonl
      frames.jsonl
      gate_map.json
      images/
        <timestamp_ns>.png
      instance_ids_gt/   # "insance_ids_gt" typo is also auto-detected
        <timestamp_ns>.png
      sequence.json
    episode_0000001/
      ...

Canonical output layout
-----------------------
OUTPUT_ROOT/
  dataset.json
  gate_geometry.json
  conversion_report.json
  splits/
    train_sequences.txt
    validation_sequences.txt
    test_sequences.txt
  sequences/
    env_000_episode_0000000/
      sequence.json
      frames.jsonl
      rgb/
        frame_000000.png
        ...
      instance_masks/
        frame_000000.png
        ...

Important source-specific facts handled by this converter
---------------------------------------------------------
1. RGB images are timestamp-named PNGs. They are preserved losslessly.
2. Source instance-mask PNG values are preserved:
       0 = background
       1..N = gate label_code
   Thus the source mask already is an instance-ID mask; no unionization occurs.
3. Source keypoints are reordered into the canonical order:
       outer_tl, outer_tr, outer_br, outer_bl,
       inner_tl, inner_tr, inner_br, inner_bl
4. Source gate-local coordinates are converted into the canonical gate frame:
       canonical +X = right when viewed from the approach side
       canonical +Y = down
       canonical +Z = through the gate from approach side
5. Visual frames (80 Hz in the supplied dataset) are aligned to the nearest
   control record (100 Hz in the supplied dataset) to attach target-gate and
   controller metadata.
6. Splits are created by whole sequence/episode, never by individual frame.

Dependencies
------------
Python >= 3.10
numpy
Pillow
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


CANONICAL_SCHEMA_VERSION = "1.0.0"
CANONICAL_SCHEMA_NAME = "uav_gate_perception_dataset"

KEYPOINT_ORDER = (
    "outer_tl",
    "outer_tr",
    "outer_br",
    "outer_bl",
    "inner_tl",
    "inner_tr",
    "inner_br",
    "inner_bl",
)

# Exact source order was verified from the supplied frame geometry:
#   src 0 = inner_tr
#   src 1 = inner_tl
#   src 2 = inner_bl
#   src 3 = inner_br
#   src 4 = outer_tr
#   src 5 = outer_tl
#   src 6 = outer_bl
#   src 7 = outer_br
SOURCE_TO_CANONICAL_KEYPOINT_INDICES = (5, 4, 7, 6, 1, 0, 3, 2)

# p_source_gate = R_SOURCE_FROM_CANONICAL_GATE @ p_canonical_gate
#
# Source gate frame inferred/validated against supplied exact 2D projections:
#   source +X = canonical +Z (through gate)
#   source +Y = canonical -X
#   source +Z = canonical -Y
R_SOURCE_FROM_CANONICAL_GATE = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)

T_SOURCE_FROM_CANONICAL_GATE = np.eye(4, dtype=np.float64)
T_SOURCE_FROM_CANONICAL_GATE[:3, :3] = R_SOURCE_FROM_CANONICAL_GATE


@dataclass(frozen=True)
class EpisodeInfo:
    env_dir: Path
    episode_dir: Path
    environment_id: int
    episode_id: int
    sequence_id: str
    source_sequence: dict[str, Any]


@dataclass(frozen=True)
class GateType:
    gate_type_id: str
    outer_width_m: float
    outer_height_m: float
    inner_width_m: float
    inner_height_m: float
    depth_m: float | None


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc


def wxyz_to_rotation(q: Iterable[float]) -> np.ndarray:
    q = np.asarray(list(q), dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"Expected quaternion [w,x,y,z], got shape {q.shape}")
    n = float(np.linalg.norm(q))
    if not math.isfinite(n) or n <= 0.0:
        raise ValueError(f"Invalid quaternion: {q.tolist()}")
    w, x, y, z = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_to_xyzw(R: np.ndarray) -> list[float]:
    """Convert a 3x3 rotation matrix to normalized quaternion [x,y,z,w]."""
    R = np.asarray(R, dtype=np.float64)
    tr = float(np.trace(R))

    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(max(0.0, 1.0 + R[1, 1] - R[0, 0] - R[2, 2])) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(max(0.0, 1.0 + R[2, 2] - R[0, 0] - R[1, 1])) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([x, y, z, w], dtype=np.float64)
    q /= np.linalg.norm(q)
    return q.tolist()


def pose_to_T(position_m: Iterable[float], orientation_wxyz: Iterable[float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = wxyz_to_rotation(orientation_wxyz)
    T[:3, 3] = np.asarray(list(position_m), dtype=np.float64)
    return T


def as_float_matrix(T: np.ndarray) -> list[list[float]]:
    return np.asarray(T, dtype=np.float64).tolist()


def canonical_gate_keypoints(
    outer_width_m: float,
    outer_height_m: float,
    inner_width_m: float,
    inner_height_m: float,
) -> dict[str, list[float]]:
    ow = outer_width_m / 2.0
    oh = outer_height_m / 2.0
    iw = inner_width_m / 2.0
    ih = inner_height_m / 2.0

    return {
        "outer_tl": [-ow, -oh, 0.0],
        "outer_tr": [ow, -oh, 0.0],
        "outer_br": [ow, oh, 0.0],
        "outer_bl": [-ow, oh, 0.0],
        "inner_tl": [-iw, -ih, 0.0],
        "inner_tr": [iw, -ih, 0.0],
        "inner_br": [iw, ih, 0.0],
        "inner_bl": [-iw, ih, 0.0],
    }


def transform_points(T: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    return (T[:3, :3] @ points_xyz.T).T + T[:3, 3]


def project_camera_points(K: np.ndarray, points_camera: np.ndarray) -> np.ndarray:
    p = np.asarray(points_camera, dtype=np.float64)
    out = np.full((len(p), 2), np.nan, dtype=np.float64)
    valid = p[:, 2] > 1e-9
    out[valid, 0] = K[0, 0] * p[valid, 0] / p[valid, 2] + K[0, 2]
    out[valid, 1] = K[1, 1] * p[valid, 1] / p[valid, 2] + K[1, 2]
    return out


def polygon_area(poly: list[list[float]] | np.ndarray) -> float:
    arr = np.asarray(poly, dtype=np.float64)
    if len(arr) < 3:
        return 0.0
    x = arr[:, 0]
    y = arr[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) * 0.5


def clip_polygon_to_rect(
    polygon: list[list[float]] | np.ndarray,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> list[list[float]]:
    """Sutherland-Hodgman clipping."""
    poly = [np.asarray(p, dtype=np.float64) for p in polygon]

    def clip(
        points: list[np.ndarray],
        inside,
        intersect,
    ) -> list[np.ndarray]:
        if not points:
            return []
        result: list[np.ndarray] = []
        prev = points[-1]
        prev_in = inside(prev)
        for cur in points:
            cur_in = inside(cur)
            if cur_in:
                if not prev_in:
                    result.append(intersect(prev, cur))
                result.append(cur)
            elif prev_in:
                result.append(intersect(prev, cur))
            prev, prev_in = cur, cur_in
        return result

    def ix(a: np.ndarray, b: np.ndarray, x: float) -> np.ndarray:
        dx = b[0] - a[0]
        if abs(dx) < 1e-12:
            return np.array([x, a[1]], dtype=np.float64)
        t = (x - a[0]) / dx
        return a + t * (b - a)

    def iy(a: np.ndarray, b: np.ndarray, y: float) -> np.ndarray:
        dy = b[1] - a[1]
        if abs(dy) < 1e-12:
            return np.array([a[0], y], dtype=np.float64)
        t = (y - a[1]) / dy
        return a + t * (b - a)

    poly = clip(poly, lambda p: p[0] >= xmin, lambda a, b: ix(a, b, xmin))
    poly = clip(poly, lambda p: p[0] <= xmax, lambda a, b: ix(a, b, xmax))
    poly = clip(poly, lambda p: p[1] >= ymin, lambda a, b: iy(a, b, ymin))
    poly = clip(poly, lambda p: p[1] <= ymax, lambda a, b: iy(a, b, ymax))
    return [p.tolist() for p in poly]


def projected_ring_areas(
    canonical_uv: np.ndarray,
    width: int,
    height: int,
) -> tuple[float, float]:
    outer = canonical_uv[:4]
    inner = canonical_uv[4:]

    if not np.all(np.isfinite(outer)) or not np.all(np.isfinite(inner)):
        return 0.0, 0.0

    full = max(0.0, polygon_area(outer) - polygon_area(inner))

    # Pixel-coordinate support is [0, W-1] x [0, H-1].
    outer_clip = clip_polygon_to_rect(outer, 0.0, 0.0, float(width - 1), float(height - 1))
    inner_clip = clip_polygon_to_rect(inner, 0.0, 0.0, float(width - 1), float(height - 1))
    in_frame = max(0.0, polygon_area(outer_clip) - polygon_area(inner_clip))
    return full, in_frame


def mask_stats(mask: np.ndarray) -> dict[int, dict[str, Any]]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {}

    labels = mask[ys, xs]
    out: dict[int, dict[str, Any]] = {}
    for label in np.unique(labels):
        label_int = int(label)
        select = labels == label
        x = xs[select]
        y = ys[select]
        out[label_int] = {
            "area": int(select.sum()),
            "bbox": [
                float(x.min()),
                float(y.min()),
                float(x.max()),
                float(y.max()),
            ],
        }
    return out


def parse_source_keypoints_px(
    value: Any,
    *,
    sequence_id: str,
    frame_index: int,
    label_code: int,
) -> np.ndarray:
    """
    Parse source keypoints_px into float64 [8,2].

    Isaac schema-v5 legitimately writes a keypoint as JSON null when that
    physical corner cannot be projected (most commonly because it is behind
    the camera). Canonical internal representation uses [nan, nan] for such
    entries so array operations remain well-defined while emitted JSON still
    writes projected_px=null.
    """
    if not isinstance(value, list) or len(value) != 8:
        raise ValueError(
            f"{sequence_id} frame {frame_index} gate {label_code}: "
            f"expected keypoints_px to be a list of length 8, got "
            f"{type(value).__name__} with value {value!r}"
        )

    out = np.full((8, 2), np.nan, dtype=np.float64)

    for i, point in enumerate(value):
        if point is None:
            continue

        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(
                f"{sequence_id} frame {frame_index} gate {label_code}: "
                f"keypoints_px[{i}] must be [x,y] or null, got {point!r}"
            )

        x, y = point
        if x is None or y is None:
            continue

        try:
            x = float(x)
            y = float(y)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{sequence_id} frame {frame_index} gate {label_code}: "
                f"keypoints_px[{i}] contains non-numeric values: {point!r}"
            ) from exc

        if not (math.isfinite(x) and math.isfinite(y)):
            continue

        out[i] = [x, y]

    return out


def keypoint_visibility_state(
    uv: np.ndarray,
    z_camera: float,
    mask: np.ndarray,
    label_code: int,
    radius_px: int,
) -> tuple[bool, bool, bool, str]:
    h, w = mask.shape

    if not math.isfinite(float(z_camera)):
        return False, False, False, "invalid"

    if z_camera <= 1e-9:
        return False, False, False, "behind_camera"

    if not np.all(np.isfinite(uv)):
        return False, False, False, "invalid"

    x, y = float(uv[0]), float(uv[1])
    in_frame = 0.0 <= x < w and 0.0 <= y < h
    if not in_frame:
        return True, False, False, "out_of_frame"

    cx = int(round(x))
    cy = int(round(y))
    x0 = max(0, cx - radius_px)
    x1 = min(w, cx + radius_px + 1)
    y0 = max(0, cy - radius_px)
    y1 = min(h, cy + radius_px + 1)

    visible = bool(np.any(mask[y0:y1, x0:x1] == label_code))
    if visible:
        return True, True, False, "visible"

    return True, False, True, "occluded"


def euclidean_norm(v: Iterable[float]) -> float:
    a = np.asarray(list(v), dtype=np.float64)
    return float(np.linalg.norm(a))


def view_angle_deg(T_camera_gate: np.ndarray) -> float | None:
    center = T_camera_gate[:3, 3]
    n = T_camera_gate[:3, 2]  # canonical +Z is gate through-direction
    center_norm = np.linalg.norm(center)
    n_norm = np.linalg.norm(n)
    if center_norm <= 1e-12 or n_norm <= 1e-12:
        return None
    cosv = abs(float(np.dot(center / center_norm, n / n_norm)))
    cosv = min(1.0, max(-1.0, cosv))
    return math.degrees(math.acos(cosv))


def find_mask_dir(episode_dir: Path) -> Path:
    candidates = (
        "instance_ids_gt",
        "insance_ids_gt",  # tolerate the spelling shown in the prompt
        "instance_id_gt",
        "instance_masks_gt",
    )
    for name in candidates:
        p = episode_dir / name
        if p.is_dir():
            return p

    fuzzy = [
        p
        for p in episode_dir.iterdir()
        if p.is_dir() and "instance" in p.name.lower() and "gt" in p.name.lower()
    ]
    if len(fuzzy) == 1:
        return fuzzy[0]

    raise FileNotFoundError(
        f"Could not find instance-mask directory under {episode_dir}. "
        f"Tried: {', '.join(candidates)}"
    )


def find_timestamp_file(directory: Path, timestamp_ns: int) -> Path:
    stem = str(int(timestamp_ns))
    for ext in (".png", ".jpg", ".jpeg"):
        p = directory / f"{stem}{ext}"
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"No file for timestamp {timestamp_ns} in {directory} "
        "(expected <timestamp_ns>.png/.jpg/.jpeg)"
    )


def transfer_file(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"

    if mode == "symlink":
        os.symlink(src.resolve(), dst)
        return "symlink"

    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy_fallback"

    raise ValueError(f"Unknown transfer mode: {mode}")


def nearest_control(
    timestamp_ns: int,
    control_timestamps: list[int],
    controls: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not controls:
        return None

    i = bisect.bisect_left(control_timestamps, timestamp_ns)
    candidates: list[int] = []
    if i < len(controls):
        candidates.append(i)
    if i > 0:
        candidates.append(i - 1)

    best = min(candidates, key=lambda j: abs(control_timestamps[j] - timestamp_ns))
    return controls[best]


def allocate_split_counts(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    raw = [n * r for r in ratios]
    counts = [math.floor(x) for x in raw]
    remaining = n - sum(counts)
    order = sorted(range(3), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in order[:remaining]:
        counts[i] += 1

    # When there are enough sequences, avoid an accidentally empty val/test split.
    if n >= 3:
        for target in (1, 2):
            if counts[target] == 0:
                donor = max(range(3), key=lambda i: counts[i])
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[target] += 1

    return tuple(counts)  # type: ignore[return-value]


def discover_episodes(input_root: Path, include_incomplete: bool) -> list[EpisodeInfo]:
    episodes: list[EpisodeInfo] = []

    for env_dir in sorted(input_root.glob("env_*")):
        if not env_dir.is_dir():
            continue

        env_meta_path = env_dir / "environment.json"
        env_meta = load_json(env_meta_path) if env_meta_path.is_file() else {}
        environment_id = int(
            env_meta.get(
                "environment_id",
                int(env_dir.name.split("_")[-1]),
            )
        )

        for episode_dir in sorted(env_dir.glob("episode_*")):
            if not episode_dir.is_dir():
                continue

            seq_path = episode_dir / "sequence.json"
            if not seq_path.is_file():
                print(f"WARNING: skipping {episode_dir}: missing sequence.json", file=sys.stderr)
                continue

            seq = load_json(seq_path)
            complete = bool(seq.get("complete", False))
            if not complete and not include_incomplete:
                print(f"INFO: skipping incomplete episode {episode_dir}", file=sys.stderr)
                continue

            episode_id = int(
                seq.get(
                    "episode_id",
                    int(episode_dir.name.split("_")[-1]),
                )
            )
            sequence_id = f"env_{environment_id:03d}_episode_{episode_id:07d}"

            required = (
                episode_dir / "frames.jsonl",
                episode_dir / "gate_map.json",
                episode_dir / "images",
            )
            missing = [str(p) for p in required if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    f"{episode_dir}: required source paths missing: {missing}"
                )
            find_mask_dir(episode_dir)

            episodes.append(
                EpisodeInfo(
                    env_dir=env_dir,
                    episode_dir=episode_dir,
                    environment_id=environment_id,
                    episode_id=episode_id,
                    sequence_id=sequence_id,
                    source_sequence=seq,
                )
            )

    return episodes


def gate_type_key(
    gate: dict[str, Any],
    depth_m: float | None,
) -> tuple[float, float, float, float, float | None]:
    outer = gate["outer_size_m"]
    inner = gate["inner_size_m"]
    return (
        float(outer[0]),
        float(outer[1]),
        float(inner[0]),
        float(inner[1]),
        None if depth_m is None else float(depth_m),
    )


def build_gate_types(
    episodes: list[EpisodeInfo],
    depth_m: float | None,
) -> tuple[dict[tuple[float, float, float, float, float | None], GateType], dict[str, Any]]:
    unique: set[tuple[float, float, float, float, float | None]] = set()

    for episode in episodes:
        gm = load_json(episode.episode_dir / "gate_map.json")
        for gate in gm.get("gates", []):
            unique.add(gate_type_key(gate, depth_m))

    ordered = sorted(
        unique,
        key=lambda x: tuple(float("-inf") if v is None else v for v in x),
    )

    registry: dict[tuple[float, float, float, float, float | None], GateType] = {}
    output_types: dict[str, Any] = {}

    for index, key in enumerate(ordered):
        ow, oh, iw, ih, depth = key
        gate_type_id = "standard_gate" if len(ordered) == 1 else f"gate_type_{index:03d}"
        gt = GateType(gate_type_id, ow, oh, iw, ih, depth)
        registry[key] = gt

        output_types[gate_type_id] = {
            "outer_width_m": ow,
            "outer_height_m": oh,
            "inner_width_m": iw,
            "inner_height_m": ih,
            "depth_m": depth,
            "keypoint_reference_surface": "gate_center_plane",
            "keypoints_gate_frame_m": canonical_gate_keypoints(ow, oh, iw, ih),
        }

    return registry, {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "gate_frame": {
            "x_axis": "right_when_viewed_from_approach_side",
            "y_axis": "down_when_viewed_from_approach_side",
            "z_axis": "through_gate_from_approach_side",
            "handedness": "right",
        },
        "gate_types": output_types,
    }


def build_split_map(
    episodes: list[EpisodeInfo],
    seed: int,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    ids = [e.sequence_id for e in episodes]
    rng = random.Random(seed)
    rng.shuffle(ids)

    n_train, n_val, n_test = allocate_split_counts(len(ids), (0.8, 0.1, 0.1))
    assert n_train + n_val + n_test == len(ids)

    split_lists = {
        "train": sorted(ids[:n_train]),
        "validation": sorted(ids[n_train : n_train + n_val]),
        "test": sorted(ids[n_train + n_val :]),
    }

    split_map: dict[str, str] = {}
    for split, seq_ids in split_lists.items():
        for sid in seq_ids:
            split_map[sid] = split

    return split_map, split_lists


def make_dataset_metadata(
    source_dataset: dict[str, Any],
    calibration: dict[str, Any],
    num_sequences: int,
    split_lists: dict[str, list[str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    K = calibration["intrinsic"]

    return {
        "schema_name": CANONICAL_SCHEMA_NAME,
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "dataset_id": args.dataset_id,
        "description": "Canonical gate-perception dataset converted from Isaac Sim collection output",
        "source": {
            "format": "isaac_drone_racer_collection_schema_v5",
            "source_schema_version": source_dataset.get("schema_version"),
            "repository_branch": source_dataset.get("repository_branch"),
            "repository_commit": source_dataset.get("repository_commit"),
            "task": source_dataset.get("task"),
            "checkpoint": source_dataset.get("checkpoint"),
            "checkpoint_sha256": source_dataset.get("checkpoint_sha256"),
            "seed": source_dataset.get("seed"),
            "physics_hz": source_dataset.get("physics_hz"),
            "control_hz": source_dataset.get("control_hz"),
            "camera_hz": source_dataset.get("camera_hz"),
        },
        "coordinate_systems": {
            "world": {
                "convention": "isaac_world_as_recorded_by_source",
                "units": "meters",
                "axis_semantics": "not_reinterpreted_by_converter",
            },
            "drone_body": {
                "convention": "isaac_body_root_as_recorded_by_source",
                "units": "meters",
            },
            "camera_optical": {
                "x_axis": "right",
                "y_axis": "down",
                "z_axis": "forward",
                "handedness": "right",
            },
            "gate_local": {
                "x_axis": "right_when_viewed_from_approach_side",
                "y_axis": "down_when_viewed_from_approach_side",
                "z_axis": "through_gate_from_approach_side",
                "handedness": "right",
            },
        },
        "transform_convention": {
            "name": "T_A_B",
            "definition": "T_A_B transforms homogeneous points expressed in frame B into frame A",
            "matrix_layout": "row_major",
            "quaternion_order_in_canonical_outputs": "xyzw",
            "translation_units": "meters",
        },
        "source_gate_frame_conversion": {
            "definition": "p_source_gate = R_source_from_canonical_gate @ p_canonical_gate",
            "R_source_from_canonical_gate": R_SOURCE_FROM_CANONICAL_GATE.tolist(),
            "validated_against_source_keypoint_projections": True,
        },
        "image_coordinate_system": {
            "origin": "top_left",
            "x_direction": "right",
            "y_direction": "down",
            "pixel_coordinate_type": "float",
            "top_left_pixel_center": [0.0, 0.0],
            "in_frame_rule": "0 <= x < image_width and 0 <= y < image_height",
            "normalized_x": "x / (image_width - 1)",
            "normalized_y": "y / (image_height - 1)",
        },
        "gate_keypoint_order": list(KEYPOINT_ORDER),
        "source_keypoint_order": [
            "inner_tr",
            "inner_tl",
            "inner_bl",
            "inner_br",
            "outer_tr",
            "outer_tl",
            "outer_bl",
            "outer_br",
        ],
        "mask_encoding": {
            "format": "PNG",
            "dtype": "uint16",
            "channels": 1,
            "background_value": 0,
            "meaning": "Each nonzero integer is the source gate label_code and maps to gates[].mask_id/track_id",
            "instance_separated": True,
            "union_mask_derivation": "instance_mask > 0",
        },
        "rgb_encoding": {
            "format": "PNG",
            "color_space": "sRGB",
            "channels": 3,
            "dtype": "uint8",
            "channel_order_on_disk": "RGB",
            "preserved_losslessly_from_source": True,
        },
        "depth_encoding": None,
        "calibration": {
            "calibration_id": calibration.get("calibration_id"),
            "width_px": int(calibration["width"]),
            "height_px": int(calibration["height"]),
            "K": K,
            "distortion": calibration.get("distortion"),
            "optical_convention": calibration.get("optical_convention"),
            "mount_convention": calibration.get("mount_convention"),
        },
        "required_modalities": {
            "rgb": True,
            "instance_mask": True,
            "frame_annotations": True,
            "camera_intrinsics": True,
            "camera_pose": True,
            "drone_pose": True,
            "gate_pose": True,
            "gate_geometry": True,
            "projected_keypoints": True,
            "keypoint_visibility": True,
        },
        "optional_modalities": {
            "dense_depth": False,
            "optical_flow": False,
            "surface_normals": False,
            "imu_like_body_kinematics": True,
            "control_actions": True,
        },
        "splitting": {
            "unit": "sequence",
            "strategy": "deterministic_random_sequence_split",
            "seed": args.split_seed,
            "train_fraction_target": 0.8,
            "validation_fraction_target": 0.1,
            "test_fraction_target": 0.1,
            "frames_from_same_sequence_may_cross_splits": False,
            "counts": {
                "train": len(split_lists["train"]),
                "validation": len(split_lists["validation"]),
                "test": len(split_lists["test"]),
                "total": num_sequences,
            },
        },
        "conversion": {
            "copy_mode_requested": args.copy_mode,
            "gate_depth_m": args.gate_depth_m,
            "keypoint_visibility_radius_px": args.keypoint_visibility_radius_px,
            "include_incomplete": args.include_incomplete,
        },
    }


def make_sequence_metadata(
    episode: EpisodeInfo,
    split: str,
    source_dataset: dict[str, Any],
    calibration: dict[str, Any],
    environment: dict[str, Any],
    gate_map: dict[str, Any],
    gate_type_registry: dict[tuple[float, float, float, float, float | None], GateType],
    depth_m: float | None,
    num_frames: int,
) -> dict[str, Any]:
    K = calibration["intrinsic"]

    body_cam = calibration.get("body_root_from_camera_optical_initial")
    T_body_camera = None
    if body_cam is not None:
        T_body_camera = as_float_matrix(
            pose_to_T(body_cam["position_m"], body_cam["orientation_wxyz"])
        )

    gates_sorted = sorted(gate_map["gates"], key=lambda g: int(g["label_code"]))

    gate_tracks = []
    for route_idx, gate in enumerate(gates_sorted):
        gt = gate_type_registry[gate_type_key(gate, depth_m)]
        T_world_gate_source = pose_to_T(
            gate["world_from_gate"]["position_m"],
            gate["world_from_gate"]["orientation_wxyz"],
        )
        T_world_gate_canonical = T_world_gate_source @ T_SOURCE_FROM_CANONICAL_GATE

        gate_tracks.append(
            {
                "track_id": gate["gate_id"],
                "source_label_code": int(gate["label_code"]),
                "gate_type_id": gt.gate_type_id,
                "route_order_index": route_idx,
                "T_world_gate": as_float_matrix(T_world_gate_canonical),
                "position_world_m": T_world_gate_canonical[:3, 3].tolist(),
                "quaternion_world_xyzw": rotation_to_xyzw(T_world_gate_canonical[:3, :3]),
                "source_prim_path": gate.get("prim_path"),
            }
        )

    return {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "sequence_id": episode.sequence_id,
        "split": split,
        "num_frames": num_frames,
        "source": {
            "environment_id": episode.environment_id,
            "episode_id": episode.episode_id,
            "source_sequence": episode.source_sequence,
        },
        "simulator": {
            "name": "Isaac Sim",
            "version": None,
            "collection_task": source_dataset.get("task"),
            "simulation_seed": source_dataset.get("seed"),
        },
        "timing": {
            "nominal_fps": source_dataset.get("camera_hz"),
            "nominal_dt_s": (
                None
                if not source_dataset.get("camera_hz")
                else 1.0 / float(source_dataset["camera_hz"])
            ),
            "control_hz": source_dataset.get("control_hz"),
            "physics_hz": source_dataset.get("physics_hz"),
        },
        "environment": environment,
        "camera": {
            "camera_id": calibration.get("calibration_id", "front_camera"),
            "width_px": int(calibration["width"]),
            "height_px": int(calibration["height"]),
            "model": "pinhole",
            "intrinsics": {
                "fx": float(K[0][0]),
                "fy": float(K[1][1]),
                "cx": float(K[0][2]),
                "cy": float(K[1][2]),
                "K": K,
            },
            "distortion": {
                "model": "brown_conrady",
                "coefficients_order": ["k1", "k2", "p1", "p2", "k3"],
                "coefficients": calibration.get("distortion"),
            },
            "T_body_camera": T_body_camera,
            "source_mount_convention": calibration.get("mount_convention"),
        },
        "course": {
            "gate_tracks": gate_tracks,
            "route_order_source": "ascending source label_code",
        },
        "domain_randomization": {
            "available_in_source_metadata": False,
            "environment_record": environment,
        },
    }


def normalize_target_gate_id(
    target_gate_id: Any,
    gate_by_label: dict[int, dict[str, Any]],
) -> tuple[int | None, str | None]:
    if target_gate_id is None:
        return None, None

    try:
        label = int(target_gate_id)
    except (TypeError, ValueError):
        return None, None

    gate = gate_by_label.get(label)
    if gate is None:
        return label, None
    return label, gate["gate_id"]


def convert_episode(
    episode: EpisodeInfo,
    output_root: Path,
    split: str,
    source_dataset: dict[str, Any],
    calibration: dict[str, Any],
    gate_type_registry: dict[tuple[float, float, float, float, float | None], GateType],
    args: argparse.Namespace,
) -> dict[str, Any]:
    ep = episode.episode_dir
    seq_out = output_root / "sequences" / episode.sequence_id
    rgb_out = seq_out / "rgb"
    mask_out = seq_out / "instance_masks"
    rgb_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    environment_path = episode.env_dir / "environment.json"
    environment = load_json(environment_path) if environment_path.is_file() else {
        "environment_id": episode.environment_id
    }

    gate_map = load_json(ep / "gate_map.json")
    gates_sorted = sorted(gate_map["gates"], key=lambda g: int(g["label_code"]))
    gate_by_label = {int(g["label_code"]): g for g in gates_sorted}
    gate_route_index = {
        int(g["label_code"]): i
        for i, g in enumerate(gates_sorted)
    }

    controls: list[dict[str, Any]] = []
    control_path = ep / "control.jsonl"
    if control_path.is_file():
        controls = list(iter_jsonl(control_path))
    control_timestamps = [int(c["timestamp_ns"]) for c in controls]

    images_dir = ep / "images"
    masks_dir = find_mask_dir(ep)

    K = np.asarray(calibration["intrinsic"], dtype=np.float64)
    width = int(calibration["width"])
    height = int(calibration["height"])

    out_frames_path = seq_out / "frames.jsonl"

    total_visible_instances = 0
    max_reprojection_error_px = 0.0
    transfer_modes_used: set[str] = set()
    converted_frames = 0
    warnings: list[str] = []

    with out_frames_path.open("w", encoding="utf-8") as fout:
        for frame_index, frame in enumerate(iter_jsonl(ep / "frames.jsonl")):
            if (
                args.max_frames_per_sequence is not None
                and frame_index >= args.max_frames_per_sequence
            ):
                break

            timestamp_ns = int(frame["timestamp_ns"])
            src_img = find_timestamp_file(images_dir, timestamp_ns)
            src_mask = find_timestamp_file(masks_dir, timestamp_ns)

            out_name = f"frame_{frame_index:06d}.png"
            dst_img = rgb_out / out_name
            dst_mask = mask_out / out_name

            # Validate/open RGB.
            with Image.open(src_img) as im:
                if im.size != (width, height):
                    raise ValueError(
                        f"{src_img}: image size {im.size} != calibration {(width, height)}"
                    )
                if im.mode not in ("RGB", "RGBA"):
                    warnings.append(f"{src_img}: RGB image mode is {im.mode!r}")

            # Load and validate instance mask.
            with Image.open(src_mask) as im:
                mask_np = np.asarray(im)
            if mask_np.ndim != 2:
                raise ValueError(f"{src_mask}: expected single-channel instance mask")
            if mask_np.shape != (height, width):
                raise ValueError(
                    f"{src_mask}: mask shape {mask_np.shape} != {(height, width)}"
                )
            if mask_np.dtype != np.uint16:
                # Preserve integer IDs, but canonicalize storage to uint16.
                if np.max(mask_np) > np.iinfo(np.uint16).max:
                    raise ValueError(f"{src_mask}: mask IDs exceed uint16 range")
                mask_np = mask_np.astype(np.uint16)
                Image.fromarray(mask_np, mode="I;16").save(dst_mask)
                transfer_modes_used.add("mask_reencoded_uint16")
            else:
                transfer_modes_used.add(transfer_file(src_mask, dst_mask, args.copy_mode))

            # Preserve RGB PNG/JPEG losslessly only when extension is PNG.
            # The supplied source uses PNG; if a future source uses JPEG, copy it
            # with its native extension rather than transcoding.
            if src_img.suffix.lower() == ".png":
                transfer_modes_used.add(transfer_file(src_img, dst_img, args.copy_mode))
                rgb_rel = f"rgb/{out_name}"
            else:
                dst_img_native = rgb_out / f"frame_{frame_index:06d}{src_img.suffix.lower()}"
                transfer_modes_used.add(transfer_file(src_img, dst_img_native, args.copy_mode))
                rgb_rel = f"rgb/{dst_img_native.name}"

            stats = mask_stats(mask_np)
            unknown_labels = sorted(set(stats) - set(gate_by_label))
            if unknown_labels:
                raise ValueError(
                    f"{src_mask}: mask contains label codes not present in gate_map.json: "
                    f"{unknown_labels}"
                )

            control = nearest_control(timestamp_ns, control_timestamps, controls)
            target_label, target_track_id = normalize_target_gate_id(
                None if control is None else control.get("target_gate_id"),
                gate_by_label,
            )
            next_track_id = None
            if target_label is not None:
                route_idx = gate_route_index.get(target_label)
                if route_idx is not None and route_idx + 1 < len(gates_sorted):
                    next_track_id = gates_sorted[route_idx + 1]["gate_id"]

            body = frame["body"]
            camera = frame["camera"]

            T_world_body = pose_to_T(
                body["position_world_m"],
                body["orientation_world_wxyz"],
            )
            T_world_camera = pose_to_T(
                camera["position_world_m"],
                camera["orientation_world_wxyz"],
            )

            vel_world = np.asarray(body["linear_velocity_world_mps"], dtype=np.float64)
            vel_body = T_world_body[:3, :3].T @ vel_world

            source_frame_gates = {
                int(g["label_code"]): g
                for g in frame.get("gates", [])
            }

            gate_records: list[dict[str, Any]] = []

            for gate_map_entry in gates_sorted:
                label = int(gate_map_entry["label_code"])
                source_gate = source_frame_gates.get(label)
                if source_gate is None:
                    # Preserve a clear missing annotation rather than inventing it.
                    warnings.append(
                        f"{episode.sequence_id} frame {frame_index}: "
                        f"missing frame gate record for label {label}"
                    )
                    continue

                gt = gate_type_registry[
                    gate_type_key(gate_map_entry, args.gate_depth_m)
                ]
                kp_gate_dict = canonical_gate_keypoints(
                    gt.outer_width_m,
                    gt.outer_height_m,
                    gt.inner_width_m,
                    gt.inner_height_m,
                )
                kp_gate = np.asarray(
                    [kp_gate_dict[name] for name in KEYPOINT_ORDER],
                    dtype=np.float64,
                )

                T_camera_gate_source = pose_to_T(
                    source_gate["position_camera_m"],
                    source_gate["orientation_camera_wxyz"],
                )
                T_camera_gate = T_camera_gate_source @ T_SOURCE_FROM_CANONICAL_GATE

                T_world_gate_source = pose_to_T(
                    gate_map_entry["world_from_gate"]["position_m"],
                    gate_map_entry["world_from_gate"]["orientation_wxyz"],
                )
                T_world_gate = T_world_gate_source @ T_SOURCE_FROM_CANONICAL_GATE

                kp_camera = transform_points(T_camera_gate, kp_gate)
                kp_world = transform_points(T_world_gate, kp_gate)

                source_kp = parse_source_keypoints_px(
                    source_gate.get("keypoints_px"),
                    sequence_id=episode.sequence_id,
                    frame_index=frame_index,
                    label_code=label,
                )
                canonical_uv = source_kp[list(SOURCE_TO_CANONICAL_KEYPOINT_INDICES)]

                # Validate that source ordering/frame conversion is still correct.
                projected_uv = project_camera_points(K, kp_camera)
                valid_proj = np.isfinite(projected_uv).all(axis=1) & np.isfinite(canonical_uv).all(axis=1)
                if np.any(valid_proj):
                    reproj = np.linalg.norm(
                        projected_uv[valid_proj] - canonical_uv[valid_proj],
                        axis=1,
                    )
                    local_max = float(np.max(reproj))
                    max_reprojection_error_px = max(max_reprojection_error_px, local_max)
                    if local_max > args.max_reprojection_error_px:
                        raise ValueError(
                            f"{episode.sequence_id} frame {frame_index} gate {label}: "
                            f"reprojection mismatch {local_max:.4f}px exceeds "
                            f"{args.max_reprojection_error_px}px. Source keypoint convention "
                            "may have changed."
                        )

                label_stats = stats.get(label)
                visible_area = 0 if label_stats is None else int(label_stats["area"])
                visible_bbox = None if label_stats is None else label_stats["bbox"]
                mask_id = label if visible_area > 0 else None
                if visible_area > 0:
                    total_visible_instances += 1

                full_area, in_frame_area = projected_ring_areas(
                    canonical_uv,
                    width,
                    height,
                )

                truncation = None
                if full_area > 1e-9:
                    truncation = min(1.0, max(0.0, 1.0 - in_frame_area / full_area))

                visible_fraction = None
                occlusion_fraction = None
                if in_frame_area > 1e-9:
                    visible_fraction = min(1.0, max(0.0, visible_area / in_frame_area))
                    occlusion_fraction = 1.0 - visible_fraction

                amodal_bbox = None
                outer_uv = canonical_uv[:4]
                if np.all(np.isfinite(outer_uv)):
                    amodal_bbox = [
                        float(np.min(outer_uv[:, 0])),
                        float(np.min(outer_uv[:, 1])),
                        float(np.max(outer_uv[:, 0])),
                        float(np.max(outer_uv[:, 1])),
                    ]

                kp_json: dict[str, Any] = {}
                for i, name in enumerate(KEYPOINT_ORDER):
                    uv = canonical_uv[i]
                    projection_valid, visible, occluded, state = keypoint_visibility_state(
                        uv,
                        float(kp_camera[i, 2]),
                        mask_np,
                        label,
                        args.keypoint_visibility_radius_px,
                    )
                    in_frame = (
                        projection_valid
                        and 0.0 <= float(uv[0]) < width
                        and 0.0 <= float(uv[1]) < height
                    )
                    norm = None
                    if projection_valid and np.all(np.isfinite(uv)):
                        norm = [
                            float(uv[0]) / float(width - 1),
                            float(uv[1]) / float(height - 1),
                        ]

                    kp_json[name] = {
                        "projected_px": (
                            None if not np.all(np.isfinite(uv)) else [float(uv[0]), float(uv[1])]
                        ),
                        "normalized": norm,
                        "projection_valid": bool(projection_valid),
                        "in_frame": bool(in_frame),
                        "visible": bool(visible),
                        "occluded": bool(occluded),
                        "visibility_state": state,
                    }

                center_camera = np.asarray(source_gate["position_camera_m"], dtype=np.float64)
                distance_camera = float(np.linalg.norm(center_camera))
                depth_camera = float(center_camera[2])

                gate_records.append(
                    {
                        "track_id": gate_map_entry["gate_id"],
                        "mask_id": mask_id,
                        "source_label_code": label,
                        "gate_type_id": gt.gate_type_id,
                        "route_order_index": gate_route_index[label],
                        "is_current_target": bool(target_label == label),
                        "pose": {
                            "T_world_gate": as_float_matrix(T_world_gate),
                            "T_camera_gate": as_float_matrix(T_camera_gate),
                            "center_camera_m": center_camera.tolist(),
                            "quaternion_camera_xyzw": rotation_to_xyzw(T_camera_gate[:3, :3]),
                            "distance_camera_m": distance_camera,
                            "depth_camera_z_m": depth_camera,
                            "view_angle_deg": view_angle_deg(T_camera_gate),
                        },
                        "visibility": {
                            "visible_pixel_area": visible_area,
                            "amodal_area_in_frame_px": float(in_frame_area),
                            "amodal_area_full_px": float(full_area),
                            "visible_fraction": visible_fraction,
                            "occlusion_fraction": occlusion_fraction,
                            "truncation_fraction": truncation,
                            "has_visible_pixels": bool(visible_area > 0),
                            "fully_visible": bool(
                                visible_area > 0
                                and (truncation is not None and truncation <= 1e-6)
                                and (occlusion_fraction is not None and occlusion_fraction <= 0.02)
                            ),
                            "partially_occluded": bool(
                                visible_area > 0
                                and occlusion_fraction is not None
                                and occlusion_fraction > 0.02
                            ),
                            "partially_out_of_frame": bool(
                                truncation is not None and truncation > 1e-6
                            ),
                            "occlusion_estimation_method": (
                                "visible_instance_pixels_vs_projected_center_plane_ring"
                            ),
                        },
                        "bounding_boxes": {
                            "visible_xyxy_px": visible_bbox,
                            "amodal_xyxy_px": amodal_bbox,
                        },
                        "keypoints_2d": kp_json,
                        "keypoints_3d_camera_m": {
                            name: kp_camera[i].tolist()
                            for i, name in enumerate(KEYPOINT_ORDER)
                        },
                        "keypoints_3d_world_m": {
                            name: kp_world[i].tolist()
                            for i, name in enumerate(KEYPOINT_ORDER)
                        },
                    }
                )

            control_json = None
            if control is not None:
                control_json = {
                    "control_frame": control.get("control_frame"),
                    "physics_step": control.get("physics_step"),
                    "timestamp_ns": control.get("timestamp_ns"),
                    "visual_minus_control_time_ns": timestamp_ns - int(control["timestamp_ns"]),
                    "target_gate_id": control.get("target_gate_id"),
                    "policy_observation": control.get("policy_observation"),
                    "requested_action": control.get("requested_action"),
                    "delayed_normalized_action": control.get("delayed_normalized_action"),
                    "applied_action": control.get("applied_action"),
                    "reward": control.get("reward"),
                    "terminated": control.get("terminated"),
                    "truncated": control.get("truncated"),
                }

            record = {
                "schema_version": CANONICAL_SCHEMA_VERSION,
                "sequence_id": episode.sequence_id,
                "frame_index": frame_index,
                "frame_id": f"{episode.sequence_id}_frame_{frame_index:06d}",
                "timestamp_ns": timestamp_ns,
                "sim_time_s": timestamp_ns / 1e9,
                "files": {
                    "rgb": rgb_rel,
                    "instance_mask": f"instance_masks/{out_name}",
                    "depth_mm": None,
                    "optical_flow": None,
                    "surface_normals": None,
                },
                "image": {
                    "width_px": width,
                    "height_px": height,
                    "source_filename": src_img.name,
                },
                "camera": {
                    "intrinsics": {
                        "fx": float(K[0, 0]),
                        "fy": float(K[1, 1]),
                        "cx": float(K[0, 2]),
                        "cy": float(K[1, 2]),
                        "K": K.tolist(),
                    },
                    "T_world_camera": as_float_matrix(T_world_camera),
                    "position_world_m": np.asarray(camera["position_world_m"], dtype=float).tolist(),
                    "quaternion_world_xyzw": rotation_to_xyzw(T_world_camera[:3, :3]),
                    "motion_from_previous": camera.get("motion_from_previous"),
                },
                "drone": {
                    "T_world_body": as_float_matrix(T_world_body),
                    "position_world_m": np.asarray(body["position_world_m"], dtype=float).tolist(),
                    "quaternion_world_xyzw": rotation_to_xyzw(T_world_body[:3, :3]),
                    "linear_velocity_world_mps": vel_world.tolist(),
                    "linear_velocity_body_mps": vel_body.tolist(),
                    "angular_velocity_body_radps": body.get("angular_velocity_body_rps"),
                    "linear_acceleration_body_mps2": body.get("linear_acceleration_body_mps2"),
                },
                "route": {
                    "current_target_track_id": target_track_id,
                    "next_target_track_id": next_track_id,
                    "current_target_route_index": (
                        None if target_label is None else gate_route_index.get(target_label)
                    ),
                },
                "control": control_json,
                "gates": gate_records,
                "frame_conditions": {
                    "gate_count_visible": len(stats),
                    "gate_count_annotated": len(gate_records),
                    "drone_speed_mps": float(np.linalg.norm(vel_world)),
                    "render_randomization": None,
                },
            }

            fout.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False))
            fout.write("\n")
            converted_frames += 1

    seq_meta = make_sequence_metadata(
        episode=episode,
        split=split,
        source_dataset=source_dataset,
        calibration=calibration,
        environment=environment,
        gate_map=gate_map,
        gate_type_registry=gate_type_registry,
        depth_m=args.gate_depth_m,
        num_frames=converted_frames,
    )
    write_json(seq_out / "sequence.json", seq_meta)

    return {
        "sequence_id": episode.sequence_id,
        "split": split,
        "frames": converted_frames,
        "visible_gate_instances": total_visible_instances,
        "max_reprojection_error_px": max_reprojection_error_px,
        "transfer_modes_used": sorted(transfer_modes_used),
        "warnings": warnings[:100],
        "warning_count": len(warnings),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert Isaac Sim collection data into the canonical gate-perception dataset."
    )
    p.add_argument(
        "input_root",
        type=Path,
        help="Path to source data root, e.g. tim0908/",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output root. Default: <input_root>_canonical",
    )
    p.add_argument(
        "--dataset-id",
        default=None,
        help="Canonical dataset ID. Default: <input folder name>_canonical_v1",
    )
    p.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Deterministic sequence-level 80/10/10 split seed (default: 42)",
    )
    p.add_argument(
        "--copy-mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
        help="How to place RGB/masks in output. hardlink avoids duplicate storage and falls back to copy.",
    )
    p.add_argument(
        "--gate-depth-m",
        type=float,
        default=0.26,
        help=(
            "Gate physical depth in meters. The supplied gate_map has outer/inner sizes "
            "but no depth; default uses project gate depth 0.26 m."
        ),
    )
    p.add_argument(
        "--keypoint-visibility-radius-px",
        type=int,
        default=3,
        help="Mask-neighborhood radius used to derive keypoint visibility (default: 3 px)",
    )
    p.add_argument(
        "--max-reprojection-error-px",
        type=float,
        default=0.5,
        help="Fail conversion if canonical/source keypoint reprojection disagrees by more than this.",
    )
    p.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include episodes whose source sequence.json has complete=false.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output directory before conversion.",
    )
    p.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Optional smoke-test limit on number of sequences.",
    )
    p.add_argument(
        "--max-frames-per-sequence",
        type=int,
        default=None,
        help="Optional smoke-test limit on frames converted per sequence.",
    )
    args = p.parse_args()

    args.input_root = args.input_root.expanduser().resolve()
    if args.output is None:
        args.output = args.input_root.with_name(args.input_root.name + "_canonical")
    args.output = args.output.expanduser().resolve()

    if args.dataset_id is None:
        args.dataset_id = args.input_root.name + "_canonical_v1"

    return args


def main() -> None:
    args = parse_args()

    input_root: Path = args.input_root
    output_root: Path = args.output

    if not input_root.is_dir():
        raise SystemExit(f"Input root does not exist or is not a directory: {input_root}")

    calibration_path = input_root / "calibration.json"
    dataset_path = input_root / "dataset.json"
    if not calibration_path.is_file():
        raise SystemExit(f"Missing {calibration_path}")
    if not dataset_path.is_file():
        raise SystemExit(f"Missing {dataset_path}")

    try:
        output_root.relative_to(input_root)
        raise SystemExit(
            "Refusing to place canonical output inside the source dataset root. "
            "Use a sibling/output directory instead."
        )
    except ValueError:
        pass

    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(
                f"Output already exists: {output_root}\n"
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_root)

    source_dataset = load_json(dataset_path)
    calibration = load_json(calibration_path)

    episodes = discover_episodes(input_root, args.include_incomplete)
    if args.max_sequences is not None:
        episodes = episodes[: args.max_sequences]

    if not episodes:
        raise SystemExit("No eligible episodes found.")

    split_map, split_lists = build_split_map(episodes, args.split_seed)

    gate_type_registry, gate_geometry = build_gate_types(
        episodes,
        args.gate_depth_m,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "splits").mkdir(parents=True, exist_ok=True)
    (output_root / "sequences").mkdir(parents=True, exist_ok=True)

    for split_name, filename in (
        ("train", "train_sequences.txt"),
        ("validation", "validation_sequences.txt"),
        ("test", "test_sequences.txt"),
    ):
        with (output_root / "splits" / filename).open("w", encoding="utf-8") as f:
            for sid in split_lists[split_name]:
                f.write(sid + "\n")

    write_json(output_root / "gate_geometry.json", gate_geometry)

    dataset_meta = make_dataset_metadata(
        source_dataset=source_dataset,
        calibration=calibration,
        num_sequences=len(episodes),
        split_lists=split_lists,
        args=args,
    )
    write_json(output_root / "dataset.json", dataset_meta)

    reports: list[dict[str, Any]] = []
    total_frames = 0
    total_visible_instances = 0
    global_max_reproj = 0.0

    print(f"Input:  {input_root}")
    print(f"Output: {output_root}")
    print(
        "Splits: "
        f"train={len(split_lists['train'])} "
        f"validation={len(split_lists['validation'])} "
        f"test={len(split_lists['test'])}"
    )

    for i, episode in enumerate(episodes, start=1):
        split = split_map[episode.sequence_id]
        print(
            f"[{i}/{len(episodes)}] {episode.sequence_id} -> {split}",
            flush=True,
        )

        report = convert_episode(
            episode=episode,
            output_root=output_root,
            split=split,
            source_dataset=source_dataset,
            calibration=calibration,
            gate_type_registry=gate_type_registry,
            args=args,
        )
        reports.append(report)

        total_frames += int(report["frames"])
        total_visible_instances += int(report["visible_gate_instances"])
        global_max_reproj = max(
            global_max_reproj,
            float(report["max_reprojection_error_px"]),
        )

        print(
            f"    frames={report['frames']} "
            f"visible_instances={report['visible_gate_instances']} "
            f"max_reproj={report['max_reprojection_error_px']:.6f}px "
            f"warnings={report['warning_count']}",
            flush=True,
        )

    report_obj = {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "dataset_id": args.dataset_id,
        "sequences": len(reports),
        "frames": total_frames,
        "visible_gate_instances": total_visible_instances,
        "max_reprojection_error_px": global_max_reproj,
        "splits": {
            k: len(v)
            for k, v in split_lists.items()
        },
        "sequence_reports": reports,
    }
    write_json(output_root / "conversion_report.json", report_obj)

    print()
    print("Conversion complete.")
    print(f"  sequences:              {len(reports)}")
    print(f"  frames:                 {total_frames}")
    print(f"  visible gate instances: {total_visible_instances}")
    print(f"  max reprojection error: {global_max_reproj:.6f} px")
    print(f"  canonical root:         {output_root}")


if __name__ == "__main__":
    main()
