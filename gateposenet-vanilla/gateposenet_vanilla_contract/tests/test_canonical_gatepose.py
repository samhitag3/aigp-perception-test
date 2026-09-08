from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from gateposenet.canonical_dataset import CanonicalGateSequenceDataset
from gateposenet.losses import GatePoseLoss
from gateposenet.model_single import GatePoseNetSingle


def _write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _make_seq(root: Path, sid: str):
    seq_dir = root / "sequences" / sid
    (seq_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (seq_dir / "instance_masks").mkdir(parents=True, exist_ok=True)
    K = [[320.0, 0.0, 320.0], [0.0, 320.0, 180.0], [0.0, 0.0, 1.0]]
    _write_json(seq_dir / "sequence.json", {
        "schema_version": "1.0.0", "sequence_id": sid,
        "timing": {"nominal_fps": 30.0},
        "camera": {"width_px": 640, "height_px": 360,
                   "intrinsics": {"K": K}},
        "course": {"gate_tracks": [
            {"track_id": "gate_0001", "gate_type_id": "standard_gate", "route_order_index": 0,
             "T_world_gate": [[1,0,0,0],[0,1,0,0],[0,0,1,5],[0,0,0,1]]},
            {"track_id": "gate_0002", "gate_type_id": "standard_gate", "route_order_index": 1,
             "T_world_gate": [[1,0,0,1],[0,1,0,0],[0,0,1,8],[0,0,0,1]]}
        ]}
    })

    frames = []
    for i in range(2):
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        img[100:260, 180:340] = 255
        cv2.imwrite(str(seq_dir / "rgb" / f"frame_{i:06d}.jpg"), img)
        m = np.zeros((360, 640), dtype=np.uint16)
        m[100:260, 180:340] = 1
        m[120:200, 400:500] = 2
        cv2.imwrite(str(seq_dir / "instance_masks" / f"frame_{i:06d}.png"), m)
        kp = {
            "outer_tl": {"projected_px": [180.0, 100.0], "visible": True},
            "outer_tr": {"projected_px": [340.0, 100.0], "visible": True},
            "outer_br": {"projected_px": [340.0, 260.0], "visible": True},
            "outer_bl": {"projected_px": [180.0, 260.0], "visible": True},
            "inner_tl": {"projected_px": [220.0, 140.0], "visible": True},
            "inner_tr": {"projected_px": [300.0, 140.0], "visible": True},
            "inner_br": {"projected_px": [300.0, 220.0], "visible": True},
            "inner_bl": {"projected_px": [220.0, 220.0], "visible": True},
        }
        frames.append({
            "schema_version": "1.0.0", "sequence_id": sid, "frame_index": i,
            "timestamp_ns": int(i * 1e9 / 30),
            "files": {"rgb": f"rgb/frame_{i:06d}.jpg",
                      "instance_mask": f"instance_masks/frame_{i:06d}.png",
                      "depth_mm": None},
            "image": {"width_px": 640, "height_px": 360},
            "camera": {"intrinsics": {"K": K},
                       "T_world_camera": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]},
            "drone": {"T_world_body": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                      "linear_velocity_world_mps": [0,0,0],
                      "angular_velocity_body_radps": [0,0,0]},
            "route": {"current_target_track_id": "gate_0001", "current_target_route_index": 0},
            "gates": [
                {"track_id": "gate_0001", "mask_id": 1, "gate_type_id": "standard_gate",
                 "route_order_index": 0, "is_current_target": True,
                 "pose": {"T_camera_gate": [[1,0,0,0],[0,1,0,0],[0,0,1,5],[0,0,0,1]]},
                 "visibility": {"visible_pixel_area": 25600, "amodal_area_full_px": 25600,
                                "visible_fraction": 1.0, "truncation_fraction": 0.0,
                                "has_visible_pixels": True},
                 "keypoints_2d": kp},
                {"track_id": "gate_0002", "mask_id": 2, "gate_type_id": "standard_gate",
                 "route_order_index": 1, "is_current_target": False,
                 "pose": {"T_camera_gate": [[1,0,0,1],[0,1,0,0],[0,0,1,8],[0,0,0,1]]},
                 "visibility": {"visible_pixel_area": 8000, "amodal_area_full_px": 8000,
                                "visible_fraction": 1.0, "truncation_fraction": 0.0,
                                "has_visible_pixels": True},
                 "keypoints_2d": kp}
            ]
        })
    (seq_dir / "frames.jsonl").write_text("\n".join(json.dumps(x) for x in frames) + "\n", encoding="utf-8")


def _make_dataset(tmp_path: Path):
    root = tmp_path / "data"
    _write_json(root / "dataset.json", {"schema_name": "uav_gate_perception_dataset", "schema_version": "1.0.0"})
    _write_json(root / "gate_geometry.json", {
        "schema_version": "1.0.0",
        "gate_types": {"standard_gate": {"keypoints_gate_frame_m": {
            "outer_tl": [-1.35,-1.35,0], "outer_tr": [1.35,-1.35,0],
            "outer_br": [1.35,1.35,0], "outer_bl": [-1.35,1.35,0],
            "inner_tl": [-0.75,-0.75,0], "inner_tr": [0.75,-0.75,0],
            "inner_br": [0.75,0.75,0], "inner_bl": [-0.75,0.75,0]
        }}}
    })
    for sid in ["seq_train", "seq_val", "seq_test"]:
        _make_seq(root, sid)
    (root / "splits").mkdir(parents=True, exist_ok=True)
    (root / "splits/train_sequences.txt").write_text("seq_train\n")
    (root / "splits/validation_sequences.txt").write_text("seq_val\n")
    (root / "splits/test_sequences.txt").write_text("seq_test\n")
    return root


def test_canonical_adapter_derives_union_and_legacy_targets(tmp_path):
    root = _make_dataset(tmp_path)
    ds = CanonicalGateSequenceDataset(root, "splits/train_sequences.txt",
                                      window=2, stride=2, size=(192, 320),
                                      nominal_focal=320.0, ego_dropout=0.0)
    sample = ds[0]
    assert sample["image"].shape == (2, 3, 192, 320)
    assert sample["seg"].shape == (2, 1, 48, 80)
    assert sample["corners_uv"].shape == (2, 4, 2)
    assert sample["ego"].shape == (2, 7)
    assert torch.allclose(sample["k_scale"], torch.ones_like(sample["k_scale"]))
    # Union contains both mask IDs, not just the current target gate.
    assert float(sample["seg"].sum()) > 0
    # Legacy x/W normalization: outer TL = (180/640, 100/360).
    assert torch.allclose(sample["corners_uv"][0, 0],
                          torch.tensor([180/640, 100/360], dtype=torch.float32), atol=1e-6)
    assert torch.allclose(sample["position"][0], torch.tensor([0.0, 0.0, 5.0]))


def test_unchanged_model_and_loss_accept_adapter_batch(tmp_path):
    root = _make_dataset(tmp_path)
    ds = CanonicalGateSequenceDataset(root, "splits/train_sequences.txt",
                                      window=2, stride=2, size=(64, 96),
                                      nominal_focal=320.0, ego_dropout=0.0)
    s = ds[0]
    batch = {k: v.unsqueeze(0) for k, v in s.items()}
    model = GatePoseNetSingle(width_factor=0.5, head_hidden=32)
    pred = model(batch["image"], batch["ego"])
    loss, logs = GatePoseLoss()(pred, batch)
    assert torch.isfinite(loss)
    assert pred["corners_uv"].shape == (1, 2, 4, 2)
    assert pred["seg_logit"].shape[-2:] == (16, 24)
    assert "total" in logs


def test_multi_source_dataset_concatenates_canonical_roots(tmp_path):
    from gateposenet.data_sources import build_canonical_dataset

    r1 = _make_dataset(tmp_path / "source_a")
    r2 = _make_dataset(tmp_path / "source_b")
    cfg = {
        "sources": [
            {"name": "a", "root": str(r1)},
            {"name": "b", "root": str(r2)},
        ],
        "train_manifest": "splits/train_sequences.txt",
        "val_manifest": "splits/validation_sequences.txt",
        "test_manifest": "splits/test_sequences.txt",
        "window": 2,
        "stride": 2,
        "val_stride": 2,
        "test_stride": 2,
        "height": 64,
        "width": 96,
        "pose_sup_max_m": 20.0,
        "nominal_focal": 320.0,
    }
    ds = build_canonical_dataset(cfg, "train", stride=2, ego_dropout=0.0)
    assert len(ds) == 2
    assert ds.source_names == ["a", "b"]
    assert ds[0]["image"].shape == (2, 3, 64, 96)
    assert ds[1]["image"].shape == (2, 3, 64, 96)
