# GatePoseNet-Canonical-MG

Temporal multi-gate network adapted to the Canonical UAV Gate Perception Dataset and Output Contract v1.0.

## AIGP camera

Authoritative pinhole intrinsics:

- source image: **640 x 360**
- `fx = fy = 320 px`
- `cx = 320 px`, `cy = 180 px`
- implied **HFoV = 90 deg**
- implied VFoV = `2*atan(360/(2*320)) ~= 58.72 deg`

The source note that calls 90 deg the VFoV is treated as mislabeled. The repository trusts the numeric intrinsics.

The model canvas is 320 x 192. A 640 x 360 source frame is resized to 320 x 180 and padded 6 px on top and bottom. This preserves pinhole geometry and gives model-space `fx=fy=160`, `cx=160`, `cy=96`.

## Raw model output contract

`forward()` returns:

```python
{
  "instances": {
    "mask_logits":          [B,T,Q,Hm,Wm],
    "object_logits":        [B,T,Q],
    "boxes":                [B,T,Q,4],        # normalized xyxy on model canvas
    "keypoints":            [B,T,Q,8,2],      # normalized model-canvas coordinates, not sigmoid-clamped
    "keypoint_logits":      [B,T,Q,8],
    "visibility_logits":    [B,T,Q,8,5],
    "pose": {
      "translation":        [B,T,Q,3],        # camera optical meters
      "rotation":           [B,T,Q,6],        # continuous 6D representation
    },
    "track_embeddings":     [B,T,Q,D]
  },
  "auxiliary": {
    "target_logits":        [B,T,Q],
    "visible_fraction":     [B,T,Q],
    "depth_log":            [B,T,Q],
    "hidden_state":         [B,C,Hf,Wf]
  }
}
```

The shared decoder is used by validation, test, and video inference. It emits Canonical Perception Model Output Contract v1.0 records and uint16 instance-ID PNGs.

## Setup

```bash
uv sync --extra tune
```

## Validate dataset

```bash
uv run python scripts/validate_dataset.py --dataset ../data/synth_large_0906
```

## Make a fixed cheap subset once

```bash
uv run python scripts/make_cheap_manifest.py \
  --dataset ../data/synth_large_0906 \
  --source splits/train_sequences.txt \
  --fraction 0.15 --seed 42 \
  --output splits/train_cheap_sequences.txt
```

## Cheap training

```bash
uv run python scripts/train.py \
  --config configs/cheap.yaml \
  --dataset ../data/synth_large_0906 \
  --train-manifest splits/train_cheap_sequences.txt \
  --val-manifest splits/validation_sequences.txt \
  --run-dir ".../runs_gpn2/gateposenet_canonical_cheap_$(date +%m%d)" \
  --device cuda
```

## Longer baseline

```bash
uv run python scripts/train.py \
  --config configs/baseline.yaml \
  --dataset ../data/perception_dataset_v1 \
  --train-manifest splits/train_sequences.txt \
  --val-manifest splits/validation_sequences.txt \
  --run-dir runs/gateposenet_canonical_baseline \
  --device cuda
```

## Optuna

```bash
uv run python scripts/tune_optuna.py \
  --config configs/optuna.yaml \
  --dataset ../data/perception_dataset_v1 \
  --train-manifest splits/train_sequences.txt \
  --val-manifest splits/validation_sequences.txt \
  --study-name gateposenet_canonical \
  --storage sqlite:///runs/gateposenet_canonical_optuna/study.db \
  --run-dir runs/gateposenet_canonical_optuna \
  --n-trials 25 --device cuda
```

## Final full-train split training

```bash
uv run python scripts/train.py \
  --config runs/gateposenet_canonical_optuna/best_config.yaml \
  --dataset ../data/perception_dataset_v1 \
  --train-manifest splits/train_sequences.txt \
  --val-manifest splits/validation_sequences.txt \
  --run-dir runs/gateposenet_canonical_final \
  --device cuda
```

## Final untouched test

```bash
uv run python scripts/evaluate.py \
  --config runs/gateposenet_canonical_optuna/best_config.yaml \
  --checkpoint runs/gateposenet_canonical_final/best.pt \
  --dataset ../data/perception_dataset_v1 \
  --split test \
  --run-dir runs/gateposenet_canonical_final/test \
  --device cuda
```

## Video inference

```bash
uv run python scripts/infer_video.py \
  --config runs/gateposenet_canonical_optuna/best_config.yaml \
  --checkpoint runs/gateposenet_canonical_final/best.pt \
  --video race_video.mp4 \
  --output outputs/race_video \
  --device cuda
```

## Export ONNX

```bash
uv run python scripts/export.py \
  --config runs/gateposenet_canonical_optuna/best_config.yaml \
  --checkpoint runs/gateposenet_canonical_final/best.pt \
  --output deploy/gateposenet_canonical.onnx
```
