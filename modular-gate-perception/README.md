# Modular Gate Perception

A three-stage perception stack for UAV gate detection using the canonical dataset/output contracts defined for this project:

1. **Temporal instance segmentation** — one independently recoverable mask per gate per frame, stored as one integer instance-ID PNG (`0=background`, `1..N=frame-local gate instances`).
2. **Temporal mask-conditioned keypoints** — each tracked gate receives an RGB+binary-mask crop history and predicts the eight fixed outer/inner gate keypoints plus visibility state.
3. **Calibrated planar pose** — OpenCV IPPE/LM PnP uses the eight predicted 2D keypoints, known gate geometry, and camera intrinsics to produce `T_camera_gate`, translation, quaternion, range/depth, reprojection error, and confidence.

The first two stages are trainable neural networks. The third is deliberately geometric: the gate dimensions and camera calibration are known, so a learned pose network is unnecessary for the baseline modular approach and would add data demand and another failure mode.

## Camera calibration

The repository treats the supplied intrinsics as authoritative:

```text
image = 640 x 360
fx = 320
fy = 320
cx = 320
cy = 180
```

Therefore:

```text
HFOV = 2 atan(640 / (2*320)) = 90.000 deg
VFOV = 2 atan(360 / (2*320)) = 58.716 deg
```

So the original `VFoV=90°` label is inconsistent with the supplied matrix; `HFOV=90°` is consistent.

Run:

```bash
uv run python scripts/inspect_camera.py
```

## Architecture

```text
video / canonical temporal RGB window
                |
                v
+--------------------------------------+
| Model 1: Temporal instance segmenter |
| ResNet-18 -> ConvGRU memory          |
| -> Transformer instance queries      |
| -> object logits / boxes / masks     |
| -> track embeddings                  |
+--------------------------------------+
                |
                | frame-local instance mask
                | 0 background, 1..N gates
                v
+--------------------------------------+
| Gate tracker                         |
| mask IoU + learned track embedding   |
| persistent track_id memory           |
+--------------------------------------+
                |
                | per-track RGB+mask history
                v
+--------------------------------------+
| Model 2: Temporal keypoint network   |
| 4-channel crop (RGB + gate mask)     |
| CNN -> ConvGRU -> 8 xy + visibility  |
+--------------------------------------+
                |
                | 8 projected/amodal points
                v
+--------------------------------------+
| Model 3: Gate pose solver            |
| planar IPPE PnP -> LM refinement     |
+--------------------------------------+
                |
                v
T_camera_gate / translation / quaternion /
distance / optical depth / reprojection error
```

### Temporal behavior

There are two distinct memories:

- **segmentation window**: the segmentation network sees the previous `T` RGB frames through a ConvGRU before decoding the current frame. This helps retain weak/partially occluded gates.
- **track/keypoint window**: persistent `track_id`s associate gate detections across frames. The keypoint model keeps a separate history of RGB+mask crops for every track.

If segmentation misses a tracked gate for a very short period, the tracker can emit a **memory-only gate** for up to two frames. Such an output has:

```json
{
  "mask_id": null,
  "source": "memory_propagated",
  "visible_area_px": 0
}
```

It does **not** falsely write old pixels into the current instance PNG. Its prior box is used only to crop the current RGB frame, append a zero-mask observation to the keypoint temporal memory, and predict keypoints/pose through the short occlusion.

## Canonical dataset input

Expected root:

```text
perception_dataset_v1/
├── dataset.json
├── gate_geometry.json
├── splits/
│   ├── train_sequences.txt
│   ├── validation_sequences.txt
│   └── test_sequences.txt
└── sequences/
    └── seq_000001/
        ├── sequence.json
        ├── frames.jsonl
        ├── rgb/
        │   └── frame_000000.jpg
        └── instance_masks/
            └── frame_000000.png
```

The stored instance PNG is a **single-channel integer image** containing all separately identifiable gate instances:

```text
0 = background
1 = gate instance A
2 = gate instance B
3 = gate instance C
...
```

A binary mask for gate `k` is generated with:

```python
binary_gate_mask = instance_mask == k
```

A union mask is generated with:

```python
union_mask = instance_mask > 0
```

The data loader reads the canonical JSONL fields described in the project contract, including `track_id`, `mask_id`, bounding boxes, projected 2D keypoints, visibility state, and camera-relative gate pose.

## Install

From the repository root:

```bash
uv sync
```

If pretrained ResNet-18 weights cannot be downloaded in your environment, change:

```yaml
pretrained_backbone: false
```

in the segmentation config.

## Point the configs at the dataset

The example configs use:

```yaml
dataset:
  root: ../data/perception_dataset_v1
```

Change this once to the actual canonical dataset root if necessary. Every model reads the same folder.

Validate the dataset before training:

```bash
uv run python -m gateperception.data.validate \
  --dataset ../data/perception_dataset_v1
```

## Training strategy

The supplied configs implement the funnel discussed for the time-constrained model search:

```text
cheap smoke training
       |
       v
full baseline
       |
       v
Optuna on promising model
       |
       v
merge tuned hyperparameters into final schedule
       |
       v
final full training on the entire fixed 80% train split
       |
       v
one-time evaluation on untouched 10% test split
```

The `validation` and `test` sequences are never used as final-training examples. "Full dataset" below means the full fixed **training split**, not train+validation+test leakage.

---

# 1. Cheap training

Use this first to verify data compatibility, loss behavior, GPU memory, and whether the architecture learns anything at all.

### Segmentation

```bash
uv run python scripts/train_segmentation.py \
  --config configs/segmentation/cheap.yaml \
  --run-dir runs/seg_cheap \
  --device cuda
```

### Keypoints

The keypoint model trains independently from GT instance masks, so it can run before segmentation training is complete.

```bash
uv run python scripts/train_keypoints.py \
  --config configs/keypoints/cheap.yaml \
  --run-dir runs/keypoints_cheap \
  --device cuda
```

Each run automatically produces:

```text
best.pt
last.pt
config.yaml
training_history.json
evaluation.json
```

---

# 2. Longer baseline training

### Segmentation

```bash
uv run python scripts/train_segmentation.py \
  --config configs/segmentation/baseline.yaml \
  --run-dir runs/seg_baseline \
  --device cuda
```

### Keypoints

```bash
uv run python scripts/train_keypoints.py \
  --config configs/keypoints/baseline.yaml \
  --run-dir runs/keypoints_baseline \
  --device cuda
```

The segmentation baseline uses a 5-frame RGB window. The keypoint baseline uses a 7-frame per-gate window.

---

# 3. Optuna tuning

The tuning configs intentionally use a limited number of sequences and fewer epochs so trials are cheap. Initialize from the baseline checkpoints to avoid repeatedly relearning basic gate features.

### Segmentation tuning

```bash
uv run python scripts/tune_segmentation_optuna.py \
  --config configs/segmentation/optuna.yaml \
  --study-dir runs/seg_tuning \
  --init-checkpoint runs/seg_baseline/best.pt \
  --device cuda \
  --n-trials 30
```

Outputs include:

```text
runs/seg_tuning/study.db
runs/seg_tuning/best_config.yaml
runs/seg_tuning/trial_XXXX/
```

### Keypoint tuning

```bash
uv run python scripts/tune_keypoints_optuna.py \
  --config configs/keypoints/optuna.yaml \
  --study-dir runs/keypoints_tuning \
  --init-checkpoint runs/keypoints_baseline/best.pt \
  --device cuda \
  --n-trials 30
```

---

# 4. Build final configs from tuned values

Do not train directly with `best_config.yaml` from Optuna because that config intentionally retains the cheap tuning sequence/epoch budget.

Merge only the tuned hyperparameters into the full final schedule:

### Segmentation

```bash
uv run python scripts/make_final_config.py \
  --base-final configs/segmentation/final.yaml \
  --tuned runs/seg_tuning/best_config.yaml \
  --kind segmentation \
  --output runs/seg_tuning/final_config.yaml
```

### Keypoints

```bash
uv run python scripts/make_final_config.py \
  --base-final configs/keypoints/final.yaml \
  --tuned runs/keypoints_tuning/best_config.yaml \
  --kind keypoints \
  --output runs/keypoints_tuning/final_config.yaml
```

---

# 5. Final full training

The final configs consume every sequence in `train_sequences.txt`, while preserving validation and test holdouts.

### Segmentation

```bash
uv run python scripts/train_segmentation.py \
  --config runs/seg_tuning/final_config.yaml \
  --init-checkpoint runs/seg_baseline/best.pt \
  --run-dir runs/seg_final \
  --device cuda
```

### Keypoints

```bash
uv run python scripts/train_keypoints.py \
  --config runs/keypoints_tuning/final_config.yaml \
  --init-checkpoint runs/keypoints_baseline/best.pt \
  --run-dir runs/keypoints_final \
  --device cuda
```

The pose stage requires no training.

---

# 6. Run the complete pipeline on the untouched test split

```bash
uv run python scripts/infer_dataset.py \
  --dataset ../data/perception_dataset_v1 \
  --split test \
  --seg-checkpoint runs/seg_final/best.pt \
  --keypoint-checkpoint runs/keypoints_final/best.pt \
  --camera-config configs/camera.yaml \
  --gate-geometry configs/gate_geometry.yaml \
  --output outputs/test_predictions \
  --device cuda
```

Then compute the canonical report:

```bash
uv run python scripts/evaluate_predictions.py \
  --dataset ../data/perception_dataset_v1 \
  --split test \
  --predictions outputs/test_predictions \
  --output outputs/test_evaluation
```

This writes:

```text
outputs/test_evaluation/
├── evaluation.json
└── per_sample_metrics.csv
```

The evaluator reports segmentation IoU/Dice/precision/recall, instance precision/recall/count accuracy, keypoint pixel/normalized/PCK metrics, pose translation/rotation/reprojection error, catastrophic failures, and runtime summary.

---

# 7. Run inference on a video

The video should correspond to the calibrated `640x360` camera. By default the script refuses a different resolution rather than silently invalidating calibration.

```bash
uv run python scripts/infer_video.py \
  --video path/to/race_video.mp4 \
  --seg-checkpoint runs/seg_final/best.pt \
  --keypoint-checkpoint runs/keypoints_final/best.pt \
  --camera-config configs/camera.yaml \
  --gate-geometry configs/gate_geometry.yaml \
  --output outputs/race_video \
  --device cuda
```

If you intentionally want the script to resize a differently sized source to the calibrated resolution:

```bash
uv run python scripts/infer_video.py \
  --video path/to/race_video.mp4 \
  --seg-checkpoint runs/seg_final/best.pt \
  --keypoint-checkpoint runs/keypoints_final/best.pt \
  --output outputs/race_video \
  --device cuda \
  --resize-input
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

The instance PNGs use the canonical output encoding:

```text
0 = background
1 = predicted gate instance 1
2 = predicted gate instance 2
...
```

The numeric `mask_id` is frame-local. Temporal identity is the separate string `track_id`.

A normal full per-instance JSON record looks like:

```json
{
  "mask_id": 2,
  "source": "model",
  "track_id": "track_0007",
  "detection_score": 0.973,
  "mask_score": 0.973,
  "box_xyxy_px": [421.3, 87.2, 1104.8, 701.4],
  "visible_area_px": 62813,
  "keypoints": {
    "outer_tl": {
      "xy_px": [428.1, 93.0],
      "confidence": 0.981,
      "visibility": "visible"
    }
  },
  "pose": {
    "T_camera_gate": [[1,0,0,0.14],[0,1,0,-0.06],[0,0,1,4.22],[0,0,0,1]],
    "translation_camera_m": [0.14, -0.06, 4.22],
    "quaternion_camera_xyzw": [0.0, 0.0, 0.0, 1.0],
    "distance_camera_m": 4.223,
    "depth_camera_z_m": 4.22,
    "mean_reprojection_error_px": 2.1,
    "confidence": 0.91
  }
}
```

## Raw training output contracts

### Segmentation model

The segmentation network does **not** train on the integer PNG directly as its prediction. It emits continuous raw values:

```python
{
  "frames": [
    {
      "object_logits":      Tensor[B,Q],
      "boxes":              Tensor[B,Q,4],
      "mask_logits":        Tensor[B,Q,H/4,W/4],
      "track_embeddings":   Tensor[B,Q,D]
    }
  ]
}
```

During training, the model returns one prediction dictionary for every time position in the window so the loss can supervise temporal consistency. Hungarian matching assigns queries to GT gate instances.

Losses:

- gate/no-gate BCE
- mask BCE
- Dice
- box L1
- temporal track embedding consistency/separation

At inference, the common decoder upsamples the per-query masks, resolves overlap, and creates the integer instance-ID PNG.

### Keypoint model

Raw output:

```python
{
  "keypoints": Tensor[B,8,2],
  "visibility_logits": Tensor[B,8,4]
}
```

The four visibility classes are:

```text
0 invalid/behind-camera
1 visible
2 occluded
3 out-of-frame
```

The coordinates are relative to the current gate crop and are intentionally not clamped to `[0,1]`, because amodal/out-of-frame corners can legitimately lie outside the crop.

## Keypoint robustness to segmentation errors

At runtime the keypoint model sees predicted segmentation masks, not perfect GT masks. Training therefore corrupts masks with random erosion/dilation and local mask dropout, while also jittering/zooming the gate crop. This reduces the train/runtime domain gap without creating a second canonical dataset.

## Augmentation

Segmentation applies one synchronized transformation to every RGB/mask pair in a temporal window:

- brightness
- contrast
- saturation
- hue shift
- gamma
- Gaussian noise
- Gaussian blur
- zoom
- translation/crop-like shift
- small rotation

RGB uses linear interpolation; the instance-ID mask always uses nearest-neighbor interpolation so integer identities are preserved.

Validation/test data are never augmented.

## Repository layout

```text
modular-gate-perception/
├── pyproject.toml
├── README.md
├── configs/
│   ├── camera.yaml
│   ├── gate_geometry.yaml
│   ├── segmentation/
│   │   ├── cheap.yaml
│   │   ├── baseline.yaml
│   │   ├── optuna.yaml
│   │   └── final.yaml
│   └── keypoints/
│       ├── cheap.yaml
│       ├── baseline.yaml
│       ├── optuna.yaml
│       └── final.yaml
├── scripts/
│   ├── train_segmentation.py
│   ├── train_keypoints.py
│   ├── tune_segmentation_optuna.py
│   ├── tune_keypoints_optuna.py
│   ├── make_final_config.py
│   ├── infer_dataset.py
│   ├── evaluate_predictions.py
│   ├── infer_video.py
│   └── inspect_camera.py
└── src/gateperception/
    ├── data/
    ├── models/
    ├── training/
    ├── inference/
    ├── geometry/
    └── utils/
```

## Recommended comparison workflow

For architecture comparison, do not compare only validation loss. Use the generated `evaluation.json` and full test evaluator, emphasizing:

1. gate instance recall / missed-gate rate
2. catastrophic-failure rate
3. instance IoU / Dice
4. visible and occluded keypoint error
5. pose translation and rotation tail error (`p95/p99`)
6. latency/FPS
7. track stability / ID switches in later tracker evaluation

A slightly lower mean IoU can be the better flight model if it has substantially fewer missed gates and catastrophic pose failures.
