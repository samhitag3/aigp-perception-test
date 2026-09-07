# GatePerceiver NN

A multi-task **deep neural network** for the canonical UAV gate-perception dataset/output contracts.

It is designed to let one source-of-truth dataset feed all four experiment stages:

1. cheap architecture screening,
2. longer baseline training,
3. Optuna tuning,
4. final full-dataset training.

## Architecture

```text
T RGB frames [B,T,3,H,W]
          +
rescaled camera K [B,T,3,3]
          +
known gate geometry [2.7,2.7,1.5,1.5,0.26]
          |
          v
   ResNet feature encoder
          |
          +---- spatial feature map ---------------------+
          |                                               |
          v                                               |
 pooled frame tokens + camera/geometry embedding          |
          |                                               |
 temporal Transformer encoder                             |
          |                                               |
          +----------> frame-conditioned memory <---------+
                              |
                     learned gate queries
                              |
                     Transformer decoder
                              |
       +----------+-----------+-----------+----------+----------+
       |          |           |           |          |          |
   objectness    mask       bbox      keypoints   pose      track emb
                                       + vis
```

### Raw training output

The model returns:

```python
{
    "pred_logits":      [B,T,Q],
    "pred_masks":       [B,T,Q,H,W],
    "pred_boxes":       [B,T,Q,4],       # normalized cx,cy,w,h
    "pred_keypoints":   [B,T,Q,8,2],     # normalized, can be outside [0,1]
    "pred_visibility":  [B,T,Q,8,4],
    "pred_translation": [B,T,Q,3],       # camera frame, meters
    "pred_rotation6d":  [B,T,Q,6],
    "track_embeddings": [B,T,Q,D]
}
```

A common Hungarian matcher/decoder converts these raw predictions into the canonical `instances[]` output.

## Camera calibration

The repository assumes the supplied camera calibration is authoritative:

```text
resolution = 640 x 360
fx = fy = 320 px
cx = 320 px
cy = 180 px
```

From those values:

```text
HFoV = 2 atan(640 / (2*320)) = 90.000 degrees
VFoV = 2 atan(360 / (2*320)) = 58.716 degrees
```

Therefore a source that labels the 90-degree field as **VFoV** is inconsistent with the intrinsics. GatePerceiver uses `K`, not the mislabeled FoV field.

## Canonical dataset expected

```text
DATASET_ROOT/
├── dataset.json
├── gate_geometry.json
├── splits/
│   ├── train_sequences.txt
│   ├── validation_sequences.txt
│   └── test_sequences.txt
└── sequences/
    └── seq_XXXXXX/
        ├── sequence.json
        ├── frames.jsonl
        ├── rgb/
        └── instance_masks/
```

The instance PNG is **not a union mask**:

```text
0 background
1 gate instance 1
2 gate instance 2
3 gate instance 3
...
```

The loader dynamically converts each integer ID into a separate binary target mask. A union mask can always be obtained with `instance_mask > 0`.

## Installation

```bash
cd gateperceiver_nn
```

```bash
deactivate
uv sync --extra tune --extra dev
```

Set your canonical Isaac-converted dataset path once:

```bash
export DATASET_ROOT=../data/synth_large_0906
```

## Validate the dataset first

```bash
uv run python scripts/validate_dataset.py \
  --dataset-root "$DATASET_ROOT"
```

This checks split leakage, referenced images/masks, instance IDs, the 8 canonical keypoints, 4x4 poses, and prints the FoV derived from the first sequence's intrinsics.

For a quick validation subset:

```bash
uv run python scripts/validate_dataset.py \
  --dataset-root "$DATASET_ROOT" \
  --max-frames 1000
```

# Training funnel

## 1. Initial cheap training

Purpose: reject clearly weak architecture/formulation ideas without spending full GPU time.

Defaults:

- 320x180 model input,
- 2-frame window,
- max 1,200 training windows,
- max 250 validation windows,
- 4 epochs,
- small ResNet18 / 128-d model.

```bash
uv run python scripts/train.py \
  --config configs/cheap.yaml \
  --dataset-root "$DATASET_ROOT" \
  --run-dir "../runs_nn/cheap_gateperceiver_$(date +%m%d)" \
  --device cuda
```

Do **not** compare the cheap run directly to final models as an accuracy benchmark. It is an architecture filter.

## 2. Longer baseline training

Purpose: establish the untuned performance of the architecture on the complete train split.

```bash
uv run python scripts/train.py \
  --config configs/baseline.yaml \
  --dataset-root "$DATASET_ROOT" \
  --run-dir "../runs_nn/baseline_gateperceiver_$(date +%m%d)" \
  --device cuda
```

Baseline defaults to a 4-frame temporal window, full 640x360 resolution, ResNet34, and 25 epochs.

## 3. Optuna tuning

Install the tuning extra if you did not already:

```bash
uv sync --extra tune
```

Run the standard 30-trial search:

```bash
uv run python scripts/tune_optuna.py \
  --config configs/optuna.yaml \
  --dataset-root "$DATASET_ROOT" \
  --run-dir "../runs_nn/optuna_gateperceiver_$(date +%m%d)" \
  --device cuda \
  --n-trials 30
```

For an even cheaper first tuning pass:

```bash
uv run python scripts/tune_optuna.py \
  --config configs/optuna.yaml \
  --dataset-root "$DATASET_ROOT" \
  --run-dir "../runs_nn/optuna_gateperceiver_$(date +%m%d)" \
  --device cuda \
  --n-trials 10
```

The tuning run writes:

```text
runs/optuna_gateperceiver/
├── trial_0000/
├── trial_0001/
├── ...
├── best_overrides.yaml
└── study_summary.yaml
```

## 4. Final full-dataset training

Apply the best Optuna hyperparameters on top of the final full-training config:

```bash
uv run python scripts/train.py \
  --config configs/final.yaml \
  --overrides "../runs_nn/optuna_gateperceiver_$(date +%m%d)/best_overrides.yaml" \
  --dataset-root "$DATASET_ROOT" \
  --run-dir "../runs_nn/final_gateperceiver_$(date +%m%d)" \
  --device cuda
```

The final config uses the entire canonical train split and 80 epochs by default.

If you need to change only the final epoch budget:

```bash
uv run python scripts/train.py \
  --config configs/final.yaml \
  --overrides "../runs_nn/optuna_gateperceiver_$(date +%m%d)/best_overrides.yaml" \
  --dataset-root "$DATASET_ROOT" \
  --run-dir "../runs_nn/final_gateperceiver_$(date +%m%d)" \
  --device cuda \
  --epochs 100
```

# Final test evaluation

Do not repeatedly use the test split during model development. Once the finalist is selected:

```bash
uv run python scripts/evaluate_checkpoint.py \
  --checkpoint "runs/final_gateperceive_$(date +%m%d)r/best.pt" \
  --dataset-root "$DATASET_ROOT" \
  --split test \
  --device cuda \
  --output "runs/final_gateperceiver_$(date +%m%d)/evaluation_test.json"
```

# Video inference

```bash
uv run python scripts/infer_video.py \
  --checkpoint "runs/final_gateperceiver_$(date +%m%d)/best.pt" \
  --video ../data/refined_target/sim0721-10/video.mp4 \
  --output "outputs_nn/final_gateperceiver_$(date +%m%d)/inference_sim0721-10.mp4" \
  --device cuda
```

Output:

```text
outputs/race_video/
├── inference.json
├── frames.jsonl
└── instance_masks/
    ├── frame_000000.png
    ├── frame_000001.png
    └── ...
```

Each output PNG uses the canonical predicted instance encoding:

```text
0 background
1 predicted gate 1
2 predicted gate 2
...
```

`frames.jsonl` maps each frame-local `mask_id` to the corresponding predicted `track_id`, confidence, bbox, eight keypoints/visibility states, and camera-relative pose.

# Why the augmentations are restricted

Photometric augmentation includes brightness, contrast, color strength, blur, Gaussian noise, and JPEG degradation. Spatial augmentation includes random crop/zoom followed by resize.

The crop transform updates:

- instance masks,
- 2D keypoints,
- bounding boxes,
- `fx, fy, cx, cy`.

Camera-relative 3D pose remains unchanged under crop/resize.

Arbitrary 2D rotation is intentionally not enabled in v1 because simply rotating pixels/keypoints while leaving the camera-frame pose untouched creates inconsistent pose supervision. If rotation is later added, it should be implemented as a proper virtual-camera transformation that updates both calibration and pose.

# Model-selection score

Validation selects checkpoints using a composite score emphasizing:

- gate recall,
- instance IoU,
- segmentation precision/recall,
- translation error,
- rotation error.

All raw metrics remain visible in `evaluation.json`; the composite score is only the model-selection objective.

# Files created by each training run

```text
runs/<run>/
├── resolved_config.yaml
├── best.pt
├── last.pt
├── training_history.csv
└── evaluation.json
```

# Tiny self-contained smoke fixture

You can verify the whole data contract without Isaac data:

```bash
uv run python scripts/make_tiny_fixture.py \
  --output /tmp/gateperceiver_fixture

uv run python scripts/validate_dataset.py \
  --dataset-root /tmp/gateperceiver_fixture
```

Then point `configs/cheap.yaml` at the fixture for loader/model debugging.
