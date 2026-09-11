from __future__ import annotations
from pathlib import Path
import copy
import hashlib
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from .augment import SynchronizedAugment, image_to_tensor
from gateperceiver.contracts import KEYPOINT_NAMES, VISIBILITY_TO_INDEX
from gateperceiver.geometry import box_xyxy_to_cxcywh, matrix_to_rotation_6d
from gateperceiver.utils.io import read_json


def _load_manifest(root: Path, split: str) -> list[str]:
    names = {"train":"train_sequences.txt", "validation":"validation_sequences.txt", "val":"validation_sequences.txt", "test":"test_sequences.txt"}
    p = root / "splits" / names[split]
    return [x.strip() for x in p.read_text().splitlines() if x.strip()]


def _select_sequence_fraction(seq_ids: list[str], fraction: float, seed: int, source_id: str) -> list[str]:
    """Deterministically choose a nested fraction of sequences.

    Ranking is SHA256(seed | source_id | sequence_id), so the result does not
    depend on manifest order. With a fixed seed, the 15% set is a strict prefix
    of the 50% set (up to rounding), which makes cheap/baseline/Optuna comparable.
    """
    fraction = float(fraction)
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"sequence_fraction must be in (0,1], got {fraction}")
    ids = sorted(set(seq_ids))
    if fraction >= 1.0 or not ids:
        return ids
    ranked = sorted(
        ids,
        key=lambda sid: hashlib.sha256(f"{seed}|{source_id}|{sid}".encode("utf-8")).digest(),
    )
    n = max(1, int(round(len(ranked) * fraction)))
    return ranked[:n]


def _frame_intrinsics(frame: dict, seq_meta: dict) -> dict:
    return copy.deepcopy(frame.get("camera", {}).get("intrinsics") or seq_meta["camera"]["intrinsics"])


def _normalize_roots(root: str | Path | list[str | Path] | tuple[str | Path, ...]) -> list[Path]:
    if isinstance(root, (str, Path)):
        roots = [Path(root)]
    else:
        roots = [Path(x) for x in root]
    if not roots:
        raise ValueError("At least one dataset root is required")
    return roots


class GateSequenceDataset(Dataset):
    def __init__(self, root: str | Path | list[str | Path], split: str, window_size: int = 1, stride: int = 1,
                 image_size: tuple[int,int] = (360,640), augment: dict | None = None,
                 max_samples: int | None = None, include_fully_occluded: bool = False,
                 sequence_fraction: float = 1.0, seed: int = 42):
        self.roots = _normalize_roots(root)
        self.root = self.roots[0]  # backwards compatibility for callers expecting .root
        self.split = split
        self.window_size = int(window_size); self.stride = int(stride)
        self.out_h, self.out_w = map(int, image_size)
        self.include_fully_occluded = include_fully_occluded
        self.augment = SynchronizedAugment(augment or {}, (self.out_h,self.out_w))
        self.sequence_fraction = float(sequence_fraction)
        self.seed = int(seed)

        metas = [read_json(r/"dataset.json") for r in self.roots]
        self.dataset_meta = metas[0] if len(metas) == 1 else {
            "schema_version": metas[0].get("schema_version", "1.0.0"),
            "dataset_id": "+".join(str(m.get("dataset_id") or r.name) for m, r in zip(metas, self.roots)),
            "sources": [
                {"dataset_id": m.get("dataset_id"), "root": str(r)}
                for m, r in zip(metas, self.roots)
            ],
        }

        geometries = [read_json(r / "gate_geometry.json") for r in self.roots]

        # Keep the first geometry for backwards compatibility.
        self.gate_geometry = geometries[0]

        # Geometry is source-specific. Different datasets may contain slightly
        # different physical gate dimensions. The model receives the correct
        # geometry vector for every individual sample.
        self.geometry_by_source: dict[str, torch.Tensor] = {}

        for root_path, geometry in zip(self.roots, geometries):
            gate_types = geometry.get("gate_types", {})

            g = (
                gate_types.get("standard_gate")
                or (next(iter(gate_types.values())) if gate_types else {})
            )

            geom_values = [
                float(g.get("outer_width_m", 2.7)),
                float(g.get("outer_height_m", 2.7)),
                float(g.get("inner_width_m", 1.5)),
                float(g.get("inner_height_m", 1.5)),
                float(g.get("depth_m", 0.26)),
            ]

            if any(v <= 0 for v in geom_values):
                raise ValueError(
                    f"Invalid gate geometry for {root_path}: {geom_values}"
                )

            self.geometry_by_source[root_path.name] = torch.tensor(
                geom_values,
                dtype=torch.float32,
            )

        self.samples = []
        self.selected_sequences: dict[str, list[str]] = {}
        for root_path, meta in zip(self.roots, metas):
            source_id = root_path.name
            all_seq_ids = _load_manifest(root_path, split)
            seq_ids = _select_sequence_fraction(all_seq_ids, self.sequence_fraction, self.seed, source_id)
            self.selected_sequences[source_id] = seq_ids
            for seq_id in seq_ids:
                seq_dir = root_path/"sequences"/seq_id
                seq_meta = read_json(seq_dir/"sequence.json")
                frames = [json.loads(x) for x in (seq_dir/"frames.jsonl").read_text().splitlines() if x.strip()]
                for end in range(self.window_size-1, len(frames), self.stride):
                    start=end-self.window_size+1
                    self.samples.append((source_id, seq_id, seq_dir, seq_meta, frames[start:end+1]))
        if max_samples is not None:
            self.samples = self.samples[:int(max_samples)]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        source_id, seq_id, seq_dir, seq_meta, frames_src = self.samples[idx]
        rgbs=[]; Ks=[]; targets=[]; frame_indices=[]; timestamps=[]
        for fs in frames_src:
            f = copy.deepcopy(fs)
            f.setdefault("camera", {})["intrinsics"] = _frame_intrinsics(f, seq_meta)
            rgb = Image.open(seq_dir/f["files"]["rgb"]).convert("RGB")
            mask_path = f["files"].get("instance_mask")
            mask = np.array(Image.open(seq_dir/mask_path), dtype=np.uint16) if mask_path else np.zeros((rgb.height,rgb.width),np.uint16)
            rgb, mask, f = self.augment(rgb, mask, f)
            rgbs.append(image_to_tensor(rgb))
            k=f["camera"]["intrinsics"]; Ks.append(torch.tensor([[k["fx"],0,k["cx"]],[0,k["fy"],k["cy"]],[0,0,1]],dtype=torch.float32))
            targets.append(self._make_target(mask, f))
            frame_indices.append(int(f["frame_index"])); timestamps.append(float(f.get("sim_time_s", f.get("timestamp_s",0.0))))
        return {
            "rgb": torch.stack(rgbs,0),
            "intrinsics": torch.stack(Ks,0),
            "gate_geometry": self.geometry_by_source[source_id].clone(),
            "targets": targets,
            "meta": {
                "source_dataset": source_id,
                "sequence_id": seq_id,
                "sequence_uid": f"{source_id}:{seq_id}",
                "frame_indices": frame_indices,
                "timestamps": timestamps,
            }
        }

    def _make_target(self, instance_mask: np.ndarray, frame: dict) -> dict:
        masks=[]; boxes=[]; keypoints=[]; visibility=[]; translations=[]; rotations6d=[]; track_ids=[]; mask_ids=[]
        H,W=instance_mask.shape
        for g in frame.get("gates", []):
            mid=g.get("mask_id")
            has_pixels = mid is not None and np.any(instance_mask==int(mid))
            if not has_pixels and not self.include_fully_occluded:
                continue
            m=(instance_mask==int(mid)) if mid is not None else np.zeros_like(instance_mask,dtype=bool)
            masks.append(torch.from_numpy(m.astype(np.float32)))
            bb=g.get("bounding_boxes",{}).get("visible_xyxy_px")
            if bb is None:
                ys,xs=np.where(m); bb=[0,0,0,0] if len(xs)==0 else [xs.min(),ys.min(),xs.max()+1,ys.max()+1]
            boxes.append(torch.tensor(bb,dtype=torch.float32))
            kp=[]; vis=[]
            for name in KEYPOINT_NAMES:
                item=g.get("keypoints_2d",{}).get(name,{})
                xy=item.get("projected_px")
                kp.append([float("nan"),float("nan")] if xy is None else [xy[0]/max(1,W-1),xy[1]/max(1,H-1)])
                vis.append(VISIBILITY_TO_INDEX.get(item.get("visibility_state","invalid"),0))
            keypoints.append(torch.tensor(kp,dtype=torch.float32)); visibility.append(torch.tensor(vis,dtype=torch.long))
            T=torch.tensor(g["pose"]["T_camera_gate"],dtype=torch.float32)
            translations.append(T[:3,3]); rotations6d.append(matrix_to_rotation_6d(T[:3,:3]))
            track_ids.append(str(g.get("track_id",""))); mask_ids.append(-1 if mid is None else int(mid))
        if masks:
            masks_t=torch.stack(masks); boxes_px=torch.stack(boxes); boxes_n=box_xyxy_to_cxcywh(boxes_px,W,H)
            kp_t=torch.stack(keypoints); vis_t=torch.stack(visibility); tr_t=torch.stack(translations); r6_t=torch.stack(rotations6d)
        else:
            masks_t=torch.zeros((0,H,W),dtype=torch.float32); boxes_n=torch.zeros((0,4)); kp_t=torch.zeros((0,8,2)); vis_t=torch.zeros((0,8),dtype=torch.long); tr_t=torch.zeros((0,3)); r6_t=torch.zeros((0,6))
        return {"masks":masks_t,"boxes":boxes_n,"keypoints":kp_t,"visibility":vis_t,"translation":tr_t,"rotation6d":r6_t,"track_ids":track_ids,"mask_ids":mask_ids,"size":torch.tensor([H,W])}


def collate_gate_batch(batch: list[dict]) -> dict:
    return {
        "rgb": torch.stack([b["rgb"] for b in batch],0),
        "intrinsics": torch.stack([b["intrinsics"] for b in batch],0),
        "gate_geometry": torch.stack([b["gate_geometry"] for b in batch],0),
        "targets": [b["targets"] for b in batch],
        "meta": [b["meta"] for b in batch],
    }
