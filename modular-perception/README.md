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

## Experiment data layout

This repository is configured for two controlled training tracks:

```text
ISAAC ONLY
  ../data/refined_isaac_0908

SYNTHETIC + ISAAC
  ../data/synth_large_0906
  ../data/refined_isaac_0908
```

The combined track reads both canonical dataset roots directly. It does **not** require copying or merging the datasets into a third folder.

### Fixed reproducibility rules

Every supplied experiment config uses:

```text
seed = 42
sequence-selection seed = 42
split unit = complete sequence / trajectory
validation = full fixed Isaac validation split
final comparison = same untouched Isaac test split
```

Training fractions are selected **per source**. The sequence IDs are sorted, deterministically permuted with seed 42, and the first `ceil(N * fraction)` sequences are selected.

Because every stage uses the same permutation:

```text
cheap 15% subset  ⊂  baseline/Optuna 50% subset  ⊂  final 100% split
```

For the combined track, selection is performed independently inside each source. Therefore the Isaac 15%/50% subsets in the combined runs are exactly the same Isaac subsets used by the Isaac-only runs. The combined runs simply add the corresponding deterministic synthetic subset.

Only the **training split** is subsampled. Validation is always 100% of the fixed Isaac validation sequences so validation metrics remain directly comparable across all runs.

### Experiment budgets

| Stage | Train split used | Epochs | Optuna trials |
|---|---:|---:|---:|
| Cheap | 15% | 8 | — |
| Baseline | 50% | 30 | — |
| Optuna | same 50% as baseline | 15 / trial | 20 |
| Final | 100% | 60 | — |

The legacy configs under `configs/segmentation/` and `configs/keypoints/` now point to the **Isaac-only** version of this protocol. Explicit two-track configs are under `configs/experiments/`.

---

# Complete CLI workflow

Run everything below from the repository root.

## 0. Environment, date tag, and output directories

```bash
uv sync

DATE=$(date +%m%d)
mkdir -p ../runs_mod ../outputs_mod

uv run python scripts/inspect_camera.py
```

With September 8, `DATE` becomes `0908`, so run/output directories end in `_0908`.

## 1. Validate both canonical datasets

```bash
uv run python -m gateperception.data.validate \
  --dataset ../data/refined_isaac_0908
```

```bash
uv run python -m gateperception.data.validate \
  --dataset ../data/synth_large_0906
```

The segmentation loader accepts canonical source images larger than 640x360 and converts them to the calibrated model/deployment resolution. RGB uses area downsampling and integer instance masks use nearest-neighbor interpolation, preserving `0,1,2,...` instance IDs.

## 2. Optional but recommended: inspect the exact deterministic subsets

### Isaac-only cheap 15%

```bash
uv run python scripts/inspect_dataset_selection.py \
  --config configs/experiments/isaac/segmentation/cheap.yaml \
  --output "../runs_mod/isaac_cheap_selection_${DATE}.json"
```

### Isaac-only baseline/Optuna 50%

```bash
uv run python scripts/inspect_dataset_selection.py \
  --config configs/experiments/isaac/segmentation/baseline.yaml \
  --output "../runs_mod/isaac_50pct_selection_${DATE}.json"
```

### Combined cheap 15% per source

```bash
uv run python scripts/inspect_dataset_selection.py \
  --config configs/experiments/combined/segmentation/cheap.yaml \
  --output "../runs_mod/combined_cheap_selection_${DATE}.json"
```

### Combined baseline/Optuna 50% per source

```bash
uv run python scripts/inspect_dataset_selection.py \
  --config configs/experiments/combined/segmentation/baseline.yaml \
  --output "../runs_mod/combined_50pct_selection_${DATE}.json"
```

Segmentation and keypoint configs use the same roots, fractions, and selection seed, so they select the same sequence subsets.

---

# Track A — Isaac-only

Training source:

```text
../data/refined_isaac_0908
```

## A1. Cheap — 15%, 8 epochs

### Segmentation

```bash
uv run python scripts/train_segmentation.py \
  --config configs/experiments/isaac/segmentation/cheap.yaml \
  --run-dir "../runs_mod/isaac_seg_cheap_${DATE}" \
  --device cuda
```

### Keypoints

```bash
uv run python scripts/train_keypoints.py \
  --config configs/experiments/isaac/keypoints/cheap.yaml \
  --run-dir "../runs_mod/isaac_keypoints_cheap_${DATE}" \
  --device cuda
```

## A2. Baseline — 50%, 30 epochs

### Segmentation

```bash
uv run python scripts/train_segmentation.py \
  --config configs/experiments/isaac/segmentation/baseline.yaml \
  --run-dir "../runs_mod/isaac_seg_baseline_${DATE}" \
  --device cuda
```

### Keypoints

```bash
uv run python scripts/train_keypoints.py \
  --config configs/experiments/isaac/keypoints/baseline.yaml \
  --run-dir "../runs_mod/isaac_keypoints_baseline_${DATE}" \
  --device cuda
```

## A3. Optuna — same 50%, 20 trials x 15 epochs

Optuna itself is seeded with 42. Every trial also trains with seed 42, so differences between trials come from the proposed hyperparameters rather than a different random training seed.

### Segmentation tuning

```bash
uv run python scripts/tune_segmentation_optuna.py \
  --config configs/experiments/isaac/segmentation/optuna.yaml \
  --study-dir "../runs_mod/isaac_seg_optuna_${DATE}" \
  --init-checkpoint "../runs_mod/isaac_seg_baseline_${DATE}/best.pt" \
  --device cuda \
  --n-trials 20
```

### Keypoint tuning

```bash
uv run python scripts/tune_keypoints_optuna.py \
  --config configs/experiments/isaac/keypoints/optuna.yaml \
  --study-dir "../runs_mod/isaac_keypoints_optuna_${DATE}" \
  --init-checkpoint "../runs_mod/isaac_keypoints_baseline_${DATE}/best.pt" \
  --device cuda \
  --n-trials 20
```

## A4. Build tuned 100%-data final configs

Do not train directly from Optuna's `best_config.yaml`; that file intentionally retains the 50% / 15-epoch tuning budget. Merge only its tuned hyperparameters into the 100% / 60-epoch final config.

### Segmentation

```bash
uv run python scripts/make_final_config.py \
  --base-final configs/experiments/isaac/segmentation/final.yaml \
  --tuned "../runs_mod/isaac_seg_optuna_${DATE}/best_config.yaml" \
  --kind segmentation \
  --output "../runs_mod/isaac_seg_optuna_${DATE}/final_config.yaml"
```

### Keypoints

```bash
uv run python scripts/make_final_config.py \
  --base-final configs/experiments/isaac/keypoints/final.yaml \
  --tuned "../runs_mod/isaac_keypoints_optuna_${DATE}/best_config.yaml" \
  --kind keypoints \
  --output "../runs_mod/isaac_keypoints_optuna_${DATE}/final_config.yaml"
```

## A5. Final — 100% train split, 60 epochs

The commands below initialize from the corresponding 30-epoch baseline checkpoint, then perform the requested **60 full-data epochs**. Both experiment tracks use this same protocol.

### Segmentation

```bash
uv run python scripts/train_segmentation.py \
  --config "../runs_mod/isaac_seg_optuna_${DATE}/final_config.yaml" \
  --init-checkpoint "../runs_mod/isaac_seg_baseline_${DATE}/best.pt" \
  --run-dir "../runs_mod/isaac_seg_final_${DATE}" \
  --device cuda
```

### Keypoints

```bash
uv run python scripts/train_keypoints.py \
  --config "../runs_mod/isaac_keypoints_optuna_${DATE}/final_config.yaml" \
  --init-checkpoint "../runs_mod/isaac_keypoints_baseline_${DATE}/best.pt" \
  --run-dir "../runs_mod/isaac_keypoints_final_${DATE}" \
  --device cuda
```

## A6. Final end-to-end inference on untouched Isaac test split

```bash
uv run python scripts/infer_dataset.py \
  --dataset ../data/refined_isaac_0908 \
  --split test \
  --seg-checkpoint "../runs_mod/isaac_seg_final_${DATE}/best.pt" \
  --keypoint-checkpoint "../runs_mod/isaac_keypoints_final_${DATE}/best.pt" \
  --camera-config configs/camera.yaml \
  --gate-geometry configs/gate_geometry.yaml \
  --output "../outputs_mod/isaac_test_predictions_${DATE}" \
  --device cuda
```

```bash
uv run python scripts/evaluate_predictions.py \
  --dataset ../data/refined_isaac_0908 \
  --split test \
  --predictions "../outputs_mod/isaac_test_predictions_${DATE}" \
  --output "../outputs_mod/isaac_test_evaluation_${DATE}"
```

---

# Track B — Synthetic + Isaac

Training sources:

```text
../data/synth_large_0906
+
../data/refined_isaac_0908
```

Validation remains the same full Isaac validation split used by Track A.

## B1. Cheap — 15% of each source, 8 epochs

### Segmentation

```bash
uv run python scripts/train_segmentation.py \
  --config configs/experiments/combined/segmentation/cheap.yaml \
  --run-dir "../runs_mod/combined_seg_cheap_${DATE}" \
  --device cuda
```

### Keypoints

```bash
uv run python scripts/train_keypoints.py \
  --config configs/experiments/combined/keypoints/cheap.yaml \
  --run-dir "../runs_mod/combined_keypoints_cheap_${DATE}" \
  --device cuda
```

## B2. Baseline — 50% of each source, 30 epochs

### Segmentation

```bash
uv run python scripts/train_segmentation.py \
  --config configs/experiments/combined/segmentation/baseline.yaml \
  --run-dir "../runs_mod/combined_seg_baseline_${DATE}" \
  --device cuda
```

### Keypoints

```bash
uv run python scripts/train_keypoints.py \
  --config configs/experiments/combined/keypoints/baseline.yaml \
  --run-dir "../runs_mod/combined_keypoints_baseline_${DATE}" \
  --device cuda
```

## B3. Optuna — same 50% of each source, 20 trials x 15 epochs

### Segmentation tuning

```bash
uv run python scripts/tune_segmentation_optuna.py \
  --config configs/experiments/combined/segmentation/optuna.yaml \
  --study-dir "../runs_mod/combined_seg_optuna_${DATE}" \
  --init-checkpoint "../runs_mod/combined_seg_baseline_${DATE}/best.pt" \
  --device cuda \
  --n-trials 20
```

### Keypoint tuning

```bash
uv run python scripts/tune_keypoints_optuna.py \
  --config configs/experiments/combined/keypoints/optuna.yaml \
  --study-dir "../runs_mod/combined_keypoints_optuna_${DATE}" \
  --init-checkpoint "../runs_mod/combined_keypoints_baseline_${DATE}/best.pt" \
  --device cuda \
  --n-trials 20
```

## B4. Build tuned 100%-data final configs

### Segmentation

```bash
uv run python scripts/make_final_config.py \
  --base-final configs/experiments/combined/segmentation/final.yaml \
  --tuned "../runs_mod/combined_seg_optuna_${DATE}/best_config.yaml" \
  --kind segmentation \
  --output "../runs_mod/combined_seg_optuna_${DATE}/final_config.yaml"
```

### Keypoints

```bash
uv run python scripts/make_final_config.py \
  --base-final configs/experiments/combined/keypoints/final.yaml \
  --tuned "../runs_mod/combined_keypoints_optuna_${DATE}/best_config.yaml" \
  --kind keypoints \
  --output "../runs_mod/combined_keypoints_optuna_${DATE}/final_config.yaml"
```

## B5. Final — 100% of both training splits, 60 epochs

### Segmentation

```bash
uv run python scripts/train_segmentation.py \
  --config "../runs_mod/combined_seg_optuna_${DATE}/final_config.yaml" \
  --init-checkpoint "../runs_mod/combined_seg_baseline_${DATE}/best.pt" \
  --run-dir "../runs_mod/combined_seg_final_${DATE}" \
  --device cuda
```

### Keypoints

```bash
uv run python scripts/train_keypoints.py \
  --config "../runs_mod/combined_keypoints_optuna_${DATE}/final_config.yaml" \
  --init-checkpoint "../runs_mod/combined_keypoints_baseline_${DATE}/best.pt" \
  --run-dir "../runs_mod/combined_keypoints_final_${DATE}" \
  --device cuda
```

## B6. Fair final comparison: combined model on the same untouched Isaac test split

```bash
uv run python scripts/infer_dataset.py \
  --dataset ../data/refined_isaac_0908 \
  --split test \
  --seg-checkpoint "../runs_mod/combined_seg_final_${DATE}/best.pt" \
  --keypoint-checkpoint "../runs_mod/combined_keypoints_final_${DATE}/best.pt" \
  --camera-config configs/camera.yaml \
  --gate-geometry configs/gate_geometry.yaml \
  --output "../outputs_mod/combined_on_isaac_test_predictions_${DATE}" \
  --device cuda
```

```bash
uv run python scripts/evaluate_predictions.py \
  --dataset ../data/refined_isaac_0908 \
  --split test \
  --predictions "../outputs_mod/combined_on_isaac_test_predictions_${DATE}" \
  --output "../outputs_mod/combined_on_isaac_test_evaluation_${DATE}"
```

The two primary final reports to compare are therefore:

```text
../outputs_mod/isaac_test_evaluation_${DATE}/evaluation.json
../outputs_mod/combined_on_isaac_test_evaluation_${DATE}/evaluation.json
```

This keeps the evaluation domain identical and measures the actual value of adding synthetic training data.

## Optional diagnostic: combined model on the synthetic test split

This is useful for diagnosing source-domain performance, but it should **not** replace the common Isaac test comparison above.

```bash
uv run python scripts/infer_dataset.py \
  --dataset ../data/synth_large_0906 \
  --split test \
  --seg-checkpoint "../runs_mod/combined_seg_final_${DATE}/best.pt" \
  --keypoint-checkpoint "../runs_mod/combined_keypoints_final_${DATE}/best.pt" \
  --camera-config configs/camera.yaml \
  --gate-geometry configs/gate_geometry.yaml \
  --output "../outputs_mod/combined_on_synth_test_predictions_${DATE}" \
  --device cuda
```

```bash
uv run python scripts/evaluate_predictions.py \
  --dataset ../data/synth_large_0906 \
  --split test \
  --predictions "../outputs_mod/combined_on_synth_test_predictions_${DATE}" \
  --output "../outputs_mod/combined_on_synth_test_evaluation_${DATE}"
```

---

# Video inference after selecting the winner

Use either the Isaac-only or combined final checkpoints. Example using the combined model:

```bash
uv run python scripts/infer_video.py \
  --video path/to/race_video.mp4 \
  --seg-checkpoint "../runs_mod/combined_seg_final_${DATE}/best.pt" \
  --keypoint-checkpoint "../runs_mod/combined_keypoints_final_${DATE}/best.pt" \
  --camera-config configs/camera.yaml \
  --gate-geometry configs/gate_geometry.yaml \
  --output "../outputs_mod/race_video_combined_${DATE}" \
  --device cuda \
  --resize-input
```

`--resize-input` is only needed when the supplied video is not already 640x360 but has the same camera geometry/aspect ratio that you intentionally want mapped into the calibrated 640x360 model space.

---

# Output conventions

Training runs are stored only under:

```text
../runs_mod/
```

Examples on September 8:

```text
../runs_mod/isaac_seg_cheap_0908/
../runs_mod/isaac_keypoints_baseline_0908/
../runs_mod/combined_seg_optuna_0908/
../runs_mod/combined_keypoints_final_0908/
```

Inference/evaluation artifacts are stored only under:

```text
../outputs_mod/
```

Examples:

```text
../outputs_mod/isaac_test_predictions_0908/
../outputs_mod/isaac_test_evaluation_0908/
../outputs_mod/combined_on_isaac_test_predictions_0908/
../outputs_mod/combined_on_isaac_test_evaluation_0908/
```

Every training run contains at least:

```text
best.pt
last.pt
config.yaml
training_history.json
evaluation.json
```

Every dataset evaluation produces:

```text
evaluation.json
per_sample_metrics.csv
```

# Reproducibility summary

For the controlled comparison in this README:

```text
seed                                  42
sequence selection seed               42
cheap training fraction               15%
cheap epochs                          8
baseline training fraction            50%
baseline epochs                       30
Optuna training fraction              same 50%
Optuna trials                         20
Optuna epochs/trial                   15
final training fraction               100%
final epochs                          60
validation                            full Isaac validation split
final comparison                      full untouched Isaac test split
split unit                            sequence, never individual frame
```

This protocol is designed so differences between the Isaac-only and Synthetic+Isaac experiments come from **training data composition**, not from different random subsets, seeds, validation domains, or test sets.
