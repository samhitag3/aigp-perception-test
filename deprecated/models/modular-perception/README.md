# Modular Gate Perception

Three-stage UAV gate perception pipeline:

1. **Temporal instance segmentation**: RGB temporal window -> one independent mask per detected gate.
2. **Temporal mask-conditioned keypoints**: per-gate RGB+mask temporal history -> 8 outer/inner keypoints plus visibility state.
3. **Geometric pose**: keypoints + calibrated camera + known gate geometry -> `T_camera_gate`, translation, rotation/quaternion, range/depth, reprojection error, confidence.

The learned stages share the same canonical dataset contract. The pose stage is geometric and does **not** require training.

## Camera used for deployment/video inference

```text
resolution = 640 x 360
fx = fy = 320
cx = 320
cy = 180
HFOV = 90.000 deg
VFOV = 58.716 deg
```

`configs/camera.yaml` is authoritative for real-video inference.

For dataset evaluation, `scripts/infer_dataset.py` uses each sequence's own stored intrinsics when available and scales them to the 640x360 inference resolution. This matters for older synthetic sequences rendered with a different camera.

---

# Experiment plan

All experiments use seed **42**.

There are two data recipes:

- `isaac`: `../data/refined_isaac_0908`
- `combined`: `../data/refined_isaac_0908` + `../data/synth_large_0906`

Training budgets:

| Stage | Training sequences | Epochs | Optuna trials |
|---|---:|---:|---:|
| Cheap | 15% | 8 | - |
| Baseline | 50% | 30 | - |
| Optuna | same 50% as baseline | 15/trial | 20 |
| Final | 100% of training split | 60 | - |

Percentages are applied **by sequence**, never by frame. For the combined recipe, the percentage is applied independently to Isaac and synthetic so one source cannot crowd out the other.

The seed-42 selection is nested and deterministic:

```text
15% subset ⊂ 50% subset ⊂ 100% training split
```

Validation and test manifests are always left complete and untouched.

---

# 0. Fresh environment setup

From the repository root:

```bash
cd ~/Desktop/model-testing/modular-gate-perception

# Keep this project on Python 3.11 for reproducibility.
uv python install 3.11
uv venv --python 3.11
uv sync
source .venv/bin/activate

python --version
uv --version
```

Set common paths and the date suffix once per shell:

```bash
export STAMP="$(date +%m%d)"
export RUNS_ROOT="../runs_mod"
export OUTPUTS_ROOT="../outputs_mod"
export ISAAC_ROOT="../data/refined_isaac_0908"
export SYNTH_ROOT="../data/synth_large_0906"
export VIEWS_ROOT="../data/modular_views"

mkdir -p "$RUNS_ROOT" "$OUTPUTS_ROOT" "$VIEWS_ROOT"
```

Check the deployment camera:

```bash
uv run python scripts/inspect_camera.py
```

Expected FOV values are approximately:

```text
HFOV = 90.000 deg
VFOV = 58.716 deg
```

---

# 1. Validate the two source datasets

```bash
uv run python -m gateperception.data.validate \
  --dataset "$ISAAC_ROOT"
```

```bash
uv run python -m gateperception.data.validate \
  --dataset "$SYNTH_ROOT"
```

The segmentation loader supports source RGB/masks larger than 640x360. It downsamples RGB with area interpolation and instance masks with nearest-neighbor interpolation, preserving integer mask IDs.

---

# 2. Create deterministic dataset views

This creates symlink-only views. It does **not** copy the images or masks.

```bash
uv run python scripts/prepare_data_views.py \
  --isaac-root "$ISAAC_ROOT" \
  --synthetic-root "$SYNTH_ROOT" \
  --output-root "$VIEWS_ROOT" \
  --seed 42 \
  --cheap-fraction 0.15 \
  --baseline-fraction 0.50
```

Created views:

```text
../data/modular_views/
├── isaac_15_seed42/
├── isaac_50_seed42/
├── isaac_100_seed42/
├── combined_15_seed42/
├── combined_50_seed42/
└── combined_100_seed42/
```

Each sequence in a view is a symlink to the original source sequence. Combined views namespace sequence IDs as `isaac__...` or `synthetic__...` to prevent collisions.

Inspect exactly how many sequences were selected:

```bash
cat "$VIEWS_ROOT/isaac_15_seed42/view.json"
cat "$VIEWS_ROOT/isaac_50_seed42/view.json"
cat "$VIEWS_ROOT/isaac_100_seed42/view.json"
cat "$VIEWS_ROOT/combined_15_seed42/view.json"
cat "$VIEWS_ROOT/combined_50_seed42/view.json"
cat "$VIEWS_ROOT/combined_100_seed42/view.json"
```

Validate all six views:

```bash
for d in \
  isaac_15_seed42 \
  isaac_50_seed42 \
  isaac_100_seed42 \
  combined_15_seed42 \
  combined_50_seed42 \
  combined_100_seed42
do
  uv run python -m gateperception.data.validate \
    --dataset "$VIEWS_ROOT/$d" || exit 1
done
```

---

# 3. ISAAC-ONLY experiments

## 3A. Cheap: 15% of training sequences, 8 epochs

Segmentation:

```bash
uv run python scripts/train_segmentation.py \
  --config configs/experiments/isaac/segmentation/cheap.yaml \
  --run-dir "$RUNS_ROOT/isaac_seg_cheap_${STAMP}" \
  --device cuda
```

Keypoints:

```bash
uv run python scripts/train_keypoints.py \
  --config configs/experiments/isaac/keypoints/cheap.yaml \
  --run-dir "$RUNS_ROOT/isaac_kp_cheap_${STAMP}" \
  --device cuda
```

## 3B. Baseline: 50% of training sequences, 30 epochs

Segmentation:

```bash
uv run python scripts/train_segmentation.py \
  --config configs/experiments/isaac/segmentation/baseline.yaml \
  --run-dir "$RUNS_ROOT/isaac_seg_baseline_${STAMP}" \
  --device cuda
```

Keypoints:

```bash
uv run python scripts/train_keypoints.py \
  --config configs/experiments/isaac/keypoints/baseline.yaml \
  --run-dir "$RUNS_ROOT/isaac_kp_baseline_${STAMP}" \
  --device cuda
```

## 3C. Optuna: same 50% subset, 20 trials x 15 epochs

Every trial uses seed 42. The Optuna TPE sampler is also seeded with 42. The baseline checkpoint is used as the identical starting point for every trial.

Segmentation:

```bash
uv run python scripts/tune_segmentation_optuna.py \
  --config configs/experiments/isaac/segmentation/optuna.yaml \
  --study-dir "$RUNS_ROOT/isaac_seg_optuna_${STAMP}" \
  --init-checkpoint "$RUNS_ROOT/isaac_seg_baseline_${STAMP}/best.pt" \
  --device cuda \
  --n-trials 20
```

Keypoints:

```bash
uv run python scripts/tune_keypoints_optuna.py \
  --config configs/experiments/isaac/keypoints/optuna.yaml \
  --study-dir "$RUNS_ROOT/isaac_kp_optuna_${STAMP}" \
  --init-checkpoint "$RUNS_ROOT/isaac_kp_baseline_${STAMP}/best.pt" \
  --device cuda \
  --n-trials 20
```

Each study writes:

```text
study.db
best_config.yaml
best_trial.json
best.pt
trial_0000/
trial_0001/
...
```

## 3D. Build tuned full-data configs

```bash
uv run python scripts/make_final_config.py \
  --base-final configs/experiments/isaac/segmentation/final.yaml \
  --tuned "$RUNS_ROOT/isaac_seg_optuna_${STAMP}/best_config.yaml" \
  --kind segmentation \
  --output "$RUNS_ROOT/isaac_seg_final_config_${STAMP}.yaml"
```

```bash
uv run python scripts/make_final_config.py \
  --base-final configs/experiments/isaac/keypoints/final.yaml \
  --tuned "$RUNS_ROOT/isaac_kp_optuna_${STAMP}/best_config.yaml" \
  --kind keypoints \
  --output "$RUNS_ROOT/isaac_kp_final_config_${STAMP}.yaml"
```

## 3E. Final: 100% of training split, 60 epochs

These final runs start fresh with the tuned hyperparameters. They do not continue the Optuna checkpoint, so this is genuinely a 60-epoch full-data final training run.

Segmentation:

```bash
uv run python scripts/train_segmentation.py \
  --config "$RUNS_ROOT/isaac_seg_final_config_${STAMP}.yaml" \
  --run-dir "$RUNS_ROOT/isaac_seg_final_${STAMP}" \
  --device cuda
```

Keypoints:

```bash
uv run python scripts/train_keypoints.py \
  --config "$RUNS_ROOT/isaac_kp_final_config_${STAMP}.yaml" \
  --run-dir "$RUNS_ROOT/isaac_kp_final_${STAMP}" \
  --device cuda
```

---

# 4. COMBINED synthetic + Isaac experiments

## 4A. Cheap: 15% from each source, 8 epochs

Segmentation:

```bash
uv run python scripts/train_segmentation.py \
  --config configs/experiments/combined/segmentation/cheap.yaml \
  --run-dir "$RUNS_ROOT/combined_seg_cheap_${STAMP}" \
  --device cuda
```

Keypoints:

```bash
uv run python scripts/train_keypoints.py \
  --config configs/experiments/combined/keypoints/cheap.yaml \
  --run-dir "$RUNS_ROOT/combined_kp_cheap_${STAMP}" \
  --device cuda
```

## 4B. Baseline: same deterministic 50% selection used by Optuna, 30 epochs

Segmentation:

```bash
uv run python scripts/train_segmentation.py \
  --config configs/experiments/combined/segmentation/baseline.yaml \
  --run-dir "$RUNS_ROOT/combined_seg_baseline_${STAMP}" \
  --device cuda
```

Keypoints:

```bash
uv run python scripts/train_keypoints.py \
  --config configs/experiments/combined/keypoints/baseline.yaml \
  --run-dir "$RUNS_ROOT/combined_kp_baseline_${STAMP}" \
  --device cuda
```

## 4C. Optuna: same 50% subset, 20 trials x 15 epochs

Segmentation:

```bash
uv run python scripts/tune_segmentation_optuna.py \
  --config configs/experiments/combined/segmentation/optuna.yaml \
  --study-dir "$RUNS_ROOT/combined_seg_optuna_${STAMP}" \
  --init-checkpoint "$RUNS_ROOT/combined_seg_baseline_${STAMP}/best.pt" \
  --device cuda \
  --n-trials 20
```

Keypoints:

```bash
uv run python scripts/tune_keypoints_optuna.py \
  --config configs/experiments/combined/keypoints/optuna.yaml \
  --study-dir "$RUNS_ROOT/combined_kp_optuna_${STAMP}" \
  --init-checkpoint "$RUNS_ROOT/combined_kp_baseline_${STAMP}/best.pt" \
  --device cuda \
  --n-trials 20
```

## 4D. Build tuned full-data configs

```bash
uv run python scripts/make_final_config.py \
  --base-final configs/experiments/combined/segmentation/final.yaml \
  --tuned "$RUNS_ROOT/combined_seg_optuna_${STAMP}/best_config.yaml" \
  --kind segmentation \
  --output "$RUNS_ROOT/combined_seg_final_config_${STAMP}.yaml"
```

```bash
uv run python scripts/make_final_config.py \
  --base-final configs/experiments/combined/keypoints/final.yaml \
  --tuned "$RUNS_ROOT/combined_kp_optuna_${STAMP}/best_config.yaml" \
  --kind keypoints \
  --output "$RUNS_ROOT/combined_kp_final_config_${STAMP}.yaml"
```

## 4E. Final: 100% of both training splits, 60 epochs

Segmentation:

```bash
uv run python scripts/train_segmentation.py \
  --config "$RUNS_ROOT/combined_seg_final_config_${STAMP}.yaml" \
  --run-dir "$RUNS_ROOT/combined_seg_final_${STAMP}" \
  --device cuda
```

Keypoints:

```bash
uv run python scripts/train_keypoints.py \
  --config "$RUNS_ROOT/combined_kp_final_config_${STAMP}.yaml" \
  --run-dir "$RUNS_ROOT/combined_kp_final_${STAMP}" \
  --device cuda
```

---

# 5. Full-pipeline evaluation commands

The training scripts already write a validation `evaluation.json` for their individual stage. The commands below evaluate the **full segmentation -> keypoints -> pose stack** on the held-out test split.

## Helper pattern

First run inference over the test sequences:

```bash
uv run python scripts/infer_dataset.py \
  --dataset DATASET_ROOT \
  --split test \
  --seg-checkpoint SEG_CHECKPOINT \
  --keypoint-checkpoint KP_CHECKPOINT \
  --camera-config configs/camera.yaml \
  --gate-geometry configs/gate_geometry.yaml \
  --output PREDICTION_DIR \
  --device cuda
```

Then evaluate:

```bash
uv run python scripts/evaluate_predictions.py \
  --dataset DATASET_ROOT \
  --split test \
  --predictions PREDICTION_DIR \
  --output EVALUATION_DIR
```

## ISAAC cheap

```bash
uv run python scripts/infer_dataset.py \
  --dataset "$VIEWS_ROOT/isaac_100_seed42" \
  --split test \
  --seg-checkpoint "$RUNS_ROOT/isaac_seg_cheap_${STAMP}/best.pt" \
  --keypoint-checkpoint "$RUNS_ROOT/isaac_kp_cheap_${STAMP}/best.pt" \
  --output "$OUTPUTS_ROOT/isaac_cheap_predictions_${STAMP}" \
  --device cuda

uv run python scripts/evaluate_predictions.py \
  --dataset "$VIEWS_ROOT/isaac_100_seed42" \
  --split test \
  --predictions "$OUTPUTS_ROOT/isaac_cheap_predictions_${STAMP}" \
  --output "$OUTPUTS_ROOT/isaac_cheap_eval_${STAMP}"
```

## ISAAC baseline

```bash
uv run python scripts/infer_dataset.py \
  --dataset "$VIEWS_ROOT/isaac_100_seed42" \
  --split test \
  --seg-checkpoint "$RUNS_ROOT/isaac_seg_baseline_${STAMP}/best.pt" \
  --keypoint-checkpoint "$RUNS_ROOT/isaac_kp_baseline_${STAMP}/best.pt" \
  --output "$OUTPUTS_ROOT/isaac_baseline_predictions_${STAMP}" \
  --device cuda

uv run python scripts/evaluate_predictions.py \
  --dataset "$VIEWS_ROOT/isaac_100_seed42" \
  --split test \
  --predictions "$OUTPUTS_ROOT/isaac_baseline_predictions_${STAMP}" \
  --output "$OUTPUTS_ROOT/isaac_baseline_eval_${STAMP}"
```

## ISAAC Optuna best trials

```bash
uv run python scripts/infer_dataset.py \
  --dataset "$VIEWS_ROOT/isaac_100_seed42" \
  --split test \
  --seg-checkpoint "$RUNS_ROOT/isaac_seg_optuna_${STAMP}/best.pt" \
  --keypoint-checkpoint "$RUNS_ROOT/isaac_kp_optuna_${STAMP}/best.pt" \
  --output "$OUTPUTS_ROOT/isaac_optuna_predictions_${STAMP}" \
  --device cuda

uv run python scripts/evaluate_predictions.py \
  --dataset "$VIEWS_ROOT/isaac_100_seed42" \
  --split test \
  --predictions "$OUTPUTS_ROOT/isaac_optuna_predictions_${STAMP}" \
  --output "$OUTPUTS_ROOT/isaac_optuna_eval_${STAMP}"
```

## ISAAC final

```bash
uv run python scripts/infer_dataset.py \
  --dataset "$VIEWS_ROOT/isaac_100_seed42" \
  --split test \
  --seg-checkpoint "$RUNS_ROOT/isaac_seg_final_${STAMP}/best.pt" \
  --keypoint-checkpoint "$RUNS_ROOT/isaac_kp_final_${STAMP}/best.pt" \
  --output "$OUTPUTS_ROOT/isaac_final_predictions_${STAMP}" \
  --device cuda

uv run python scripts/evaluate_predictions.py \
  --dataset "$VIEWS_ROOT/isaac_100_seed42" \
  --split test \
  --predictions "$OUTPUTS_ROOT/isaac_final_predictions_${STAMP}" \
  --output "$OUTPUTS_ROOT/isaac_final_eval_${STAMP}"
```

## Combined cheap

```bash
uv run python scripts/infer_dataset.py \
  --dataset "$VIEWS_ROOT/combined_100_seed42" \
  --split test \
  --seg-checkpoint "$RUNS_ROOT/combined_seg_cheap_${STAMP}/best.pt" \
  --keypoint-checkpoint "$RUNS_ROOT/combined_kp_cheap_${STAMP}/best.pt" \
  --output "$OUTPUTS_ROOT/combined_cheap_predictions_${STAMP}" \
  --device cuda

uv run python scripts/evaluate_predictions.py \
  --dataset "$VIEWS_ROOT/combined_100_seed42" \
  --split test \
  --predictions "$OUTPUTS_ROOT/combined_cheap_predictions_${STAMP}" \
  --output "$OUTPUTS_ROOT/combined_cheap_eval_${STAMP}"
```

## Combined baseline

```bash
uv run python scripts/infer_dataset.py \
  --dataset "$VIEWS_ROOT/combined_100_seed42" \
  --split test \
  --seg-checkpoint "$RUNS_ROOT/combined_seg_baseline_${STAMP}/best.pt" \
  --keypoint-checkpoint "$RUNS_ROOT/combined_kp_baseline_${STAMP}/best.pt" \
  --output "$OUTPUTS_ROOT/combined_baseline_predictions_${STAMP}" \
  --device cuda

uv run python scripts/evaluate_predictions.py \
  --dataset "$VIEWS_ROOT/combined_100_seed42" \
  --split test \
  --predictions "$OUTPUTS_ROOT/combined_baseline_predictions_${STAMP}" \
  --output "$OUTPUTS_ROOT/combined_baseline_eval_${STAMP}"
```

## Combined Optuna best trials

```bash
uv run python scripts/infer_dataset.py \
  --dataset "$VIEWS_ROOT/combined_100_seed42" \
  --split test \
  --seg-checkpoint "$RUNS_ROOT/combined_seg_optuna_${STAMP}/best.pt" \
  --keypoint-checkpoint "$RUNS_ROOT/combined_kp_optuna_${STAMP}/best.pt" \
  --output "$OUTPUTS_ROOT/combined_optuna_predictions_${STAMP}" \
  --device cuda

uv run python scripts/evaluate_predictions.py \
  --dataset "$VIEWS_ROOT/combined_100_seed42" \
  --split test \
  --predictions "$OUTPUTS_ROOT/combined_optuna_predictions_${STAMP}" \
  --output "$OUTPUTS_ROOT/combined_optuna_eval_${STAMP}"
```

## Combined final aggregate test

```bash
uv run python scripts/infer_dataset.py \
  --dataset "$VIEWS_ROOT/combined_100_seed42" \
  --split test \
  --seg-checkpoint "$RUNS_ROOT/combined_seg_final_${STAMP}/best.pt" \
  --keypoint-checkpoint "$RUNS_ROOT/combined_kp_final_${STAMP}/best.pt" \
  --output "$OUTPUTS_ROOT/combined_final_predictions_${STAMP}" \
  --device cuda

uv run python scripts/evaluate_predictions.py \
  --dataset "$VIEWS_ROOT/combined_100_seed42" \
  --split test \
  --predictions "$OUTPUTS_ROOT/combined_final_predictions_${STAMP}" \
  --output "$OUTPUTS_ROOT/combined_final_eval_${STAMP}"
```

The evaluator writes:

```text
EVALUATION_DIR/
├── evaluation.json
└── per_sample_metrics.csv
```

---

# 6. Recommended domain-specific evaluation for the combined final model

The combined aggregate score is useful, but do not stop there. Evaluate the same final model separately on Isaac and synthetic to see whether one domain is dominating.

## Combined final model -> Isaac-only test

```bash
uv run python scripts/infer_dataset.py \
  --dataset "$ISAAC_ROOT" \
  --split test \
  --seg-checkpoint "$RUNS_ROOT/combined_seg_final_${STAMP}/best.pt" \
  --keypoint-checkpoint "$RUNS_ROOT/combined_kp_final_${STAMP}/best.pt" \
  --output "$OUTPUTS_ROOT/combined_final_on_isaac_predictions_${STAMP}" \
  --device cuda

uv run python scripts/evaluate_predictions.py \
  --dataset "$ISAAC_ROOT" \
  --split test \
  --predictions "$OUTPUTS_ROOT/combined_final_on_isaac_predictions_${STAMP}" \
  --output "$OUTPUTS_ROOT/combined_final_on_isaac_eval_${STAMP}"
```

## Combined final model -> synthetic-only test

```bash
uv run python scripts/infer_dataset.py \
  --dataset "$SYNTH_ROOT" \
  --split test \
  --seg-checkpoint "$RUNS_ROOT/combined_seg_final_${STAMP}/best.pt" \
  --keypoint-checkpoint "$RUNS_ROOT/combined_kp_final_${STAMP}/best.pt" \
  --output "$OUTPUTS_ROOT/combined_final_on_synth_predictions_${STAMP}" \
  --device cuda

uv run python scripts/evaluate_predictions.py \
  --dataset "$SYNTH_ROOT" \
  --split test \
  --predictions "$OUTPUTS_ROOT/combined_final_on_synth_predictions_${STAMP}" \
  --output "$OUTPUTS_ROOT/combined_final_on_synth_eval_${STAMP}"
```

---

# 7. Video -> annotated output video

`infer_video.py` now writes all canonical machine-readable outputs **and** a video with different stable colors for each tracked gate mask.

The overlay includes:

- translucent, differently colored per-gate masks
- persistent `track_id`
- detection confidence
- bounding box
- 8 predicted keypoints
- depth and range when PnP succeeds
- number of visible and memory-propagated gates
- per-frame latency and FPS

For a calibrated 640x360 input video:

## ISAAC-only final model

```bash
export VIDEO="/path/to/input.mp4"

uv run python scripts/infer_video.py \
  --video "$VIDEO" \
  --seg-checkpoint "$RUNS_ROOT/isaac_seg_final_${STAMP}/best.pt" \
  --keypoint-checkpoint "$RUNS_ROOT/isaac_kp_final_${STAMP}/best.pt" \
  --camera-config configs/camera.yaml \
  --gate-geometry configs/gate_geometry.yaml \
  --output "$OUTPUTS_ROOT/isaac_video_${STAMP}" \
  --device cuda
```

Output video:

```text
../outputs_mod/isaac_video_MMDD/overlay.mp4
```

## Combined final model

```bash
uv run python scripts/infer_video.py \
  --video "$VIDEO" \
  --seg-checkpoint "$RUNS_ROOT/combined_seg_final_${STAMP}/best.pt" \
  --keypoint-checkpoint "$RUNS_ROOT/combined_kp_final_${STAMP}/best.pt" \
  --camera-config configs/camera.yaml \
  --gate-geometry configs/gate_geometry.yaml \
  --output "$OUTPUTS_ROOT/combined_video_${STAMP}" \
  --device cuda
```

Output video:

```text
../outputs_mod/combined_video_MMDD/overlay.mp4
```

If the input video is not 640x360 and you intentionally want it resized to the calibrated model space, add:

```text
--resize-input
```

Example:

```bash
uv run python scripts/infer_video.py \
  --video "$VIDEO" \
  --seg-checkpoint "$RUNS_ROOT/combined_seg_final_${STAMP}/best.pt" \
  --keypoint-checkpoint "$RUNS_ROOT/combined_kp_final_${STAMP}/best.pt" \
  --output "$OUTPUTS_ROOT/combined_video_${STAMP}" \
  --device cuda \
  --resize-input
```

Only use `--resize-input` for videos whose imaging geometry is intended to represent the same calibrated camera after resizing. Arbitrarily resizing video from a different FOV does not make its intrinsics equal to the deployment camera.

The complete video inference output is:

```text
../outputs_mod/combined_video_MMDD/
├── inference.json
├── frames.jsonl
├── overlay.mp4
└── instance_masks/
    ├── frame_000000.png
    ├── frame_000001.png
    └── ...
```

Each instance-mask PNG is `uint16`:

```text
0 = background
1 = detected gate instance 1
2 = detected gate instance 2
3 = detected gate instance 3
...
```

---

# 8. Suggested run order

Do not run all stages blindly. The efficient order is:

```text
validate data
    ↓
create deterministic views
    ↓
ISAAC cheap + combined cheap
    ↓
evaluate both
    ↓
ISAAC baseline + combined baseline
    ↓
evaluate both
    ↓
Optuna only for recipes still worth pursuing
    ↓
60-epoch final training
    ↓
combined/Isaac held-out tests
    ↓
real-video overlay testing
```

If cheap combined is clearly better than cheap Isaac-only, you can still run both baselines for a fair comparison, but you do not need to burn 20 Optuna trials on an approach that is obviously behind.

---

# 9. Reproducibility details

- all configs: `seed: 42`
- data subset ordering: stable SHA256 ordering using seed 42
- train/validation/test split: inherited from source dataset; never frame-randomized
- 15% and 50% selection: sequence-level only
- baseline and Optuna: exact same 50% training subset
- Optuna sampler: seeded TPE with seed 42
- every Optuna trial: model/data seed reset to 42
- DataLoader train shuffling: seeded PyTorch generator
- validation/test: no training augmentation
- source datasets remain unchanged; views are symlinks only

---

# 10. Important resolution/intrinsics behavior

The segmentation network always trains/infer at 640x360.

If a source frame is 1280x720, the segmentation loader converts:

```text
RGB:       1280x720 -> 640x360 using area interpolation
instance:  1280x720 -> 640x360 using nearest-neighbor interpolation
```

Nearest-neighbor interpolation preserves instance IDs.

The keypoint network trains on per-gate crops and therefore can consume canonical source sequences at different source resolutions without needing duplicate datasets.

For **dataset pose evaluation**, sequence-specific camera intrinsics are used and scaled consistently to the 640x360 prediction image.

For **real video**, `configs/camera.yaml` is used:

```text
K = [[320,   0, 320],
     [  0, 320, 180],
     [  0,   0,   1]]
```

---

# 11. Core outputs from each run

Each learned-stage training run contains at least:

```text
../runs_mod/<run_name>_MMDD/
├── config.yaml
├── best.pt
├── last.pt
├── training_history.json
└── evaluation.json
```

Each Optuna study additionally contains:

```text
study.db
best_config.yaml
best_trial.json
best.pt
trial_XXXX/
```

Full-pipeline evaluation contains:

```text
../outputs_mod/<name>_MMDD/
├── evaluation.json
└── per_sample_metrics.csv
```

Use `evaluation.json` for model ranking and `per_sample_metrics.csv` for failure analysis versus distance, occlusion, truncation, viewing angle, etc.
