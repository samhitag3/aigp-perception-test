# GatePerceiver NN

A multi-task temporal deep neural network for the canonical UAV gate-perception dataset/output contracts.

This repository supports the full comparison funnel with **one or multiple canonical dataset roots** and deterministic sequence-level sampling:

1. **cheap**: 15% of training sequences, 8 epochs,
2. **baseline**: 50% of training sequences, 30 epochs,
3. **Optuna**: the exact same 50% sequence subset as baseline, 20 trials × 15 epochs,
4. **final**: 100% of training sequences, 60 epochs.

All random behavior uses **seed 42**. Train subsets are selected at the **sequence level**, never frame-by-frame. Validation and test always use 100% of their fixed canonical sequence manifests so comparisons use the same evaluation data.

The 15% subset is nested inside the 50% subset because sequences are deterministically ranked using seed 42. Baseline and every Optuna trial therefore use the exact same 50% training sequences for a given dataset combination.

## Architecture

```text
T RGB frames [B,T,3,H,W]
          +
rescaled camera K [B,T,3,3]
          +
known gate geometry [2.7,2.7,1.5,1.5,0.26]
          |
          v
   ResNet34 feature encoder
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

Raw training output:

```python
{
    "pred_logits":      [B,T,Q],
    "pred_masks":       [B,T,Q,H,W],
    "pred_boxes":       [B,T,Q,4],
    "pred_keypoints":   [B,T,Q,8,2],
    "pred_visibility":  [B,T,Q,8,4],
    "pred_translation": [B,T,Q,3],
    "pred_rotation6d":  [B,T,Q,6],
    "track_embeddings": [B,T,Q,D]
}
```

## Camera calibration

The supplied intrinsic matrix is authoritative:

```text
resolution = 640 x 360
fx = fy = 320 px
cx = 320 px
cy = 180 px
```

This implies:

```text
HFoV = 90.000 degrees
VFoV = 58.716 degrees
```

The external `VFoV = 90°` label is therefore treated as a metadata labeling error. The network uses `K`, not that FoV label.

## Canonical dataset format

Each dataset root must independently follow:

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

Instance masks are single-channel integer PNGs:

```text
0 = background
1 = gate instance 1
2 = gate instance 2
3 = gate instance 3
...
```

The loader dynamically converts each integer ID into a separate binary target mask. Union segmentation is derivable with `instance_mask > 0`.

# Installation

From the repository root:

```bash
cd ~/Desktop/model-testing/gateperceiver_nn

deactivate 2>/dev/null || true
uv sync --extra tune --extra dev
```

If an unrelated virtual environment such as `(synth-data-gen)` is active, `uv` will normally ignore it, but deactivating it avoids confusion.

# Dataset paths and output naming

For the September 8 datasets:

```bash
export ISAAC_ROOT="../data/refined_isaac_0908"
export SYNTH_ROOT="../data/synth_large_0906"

export RUNS_ROOT="../runs_nn"
export OUTPUTS_ROOT="../outputs_nn"

export DATE="$(date +%m%d)"

mkdir -p "$RUNS_ROOT" "$OUTPUTS_ROOT"
```

On September 8, `DATE` evaluates to `0908`, producing names such as:

```text
../runs_nn/isaac_cheap_0908
../runs_nn/combined_baseline_0908
../outputs_nn/isaac_final_0908
../outputs_nn/combined_final_0908
```

# Validate both source datasets

Validate Isaac:

```bash
uv run python scripts/validate_dataset.py \
  --dataset-root "$ISAAC_ROOT"
```

Validate synthetic:

```bash
uv run python scripts/validate_dataset.py \
  --dataset-root "$SYNTH_ROOT"
```

Optional fast validator smoke checks:

```bash
uv run python scripts/validate_dataset.py \
  --dataset-root "$ISAAC_ROOT" \
  --max-frames 1000

uv run python scripts/validate_dataset.py \
  --dataset-root "$SYNTH_ROOT" \
  --max-frames 1000
```

# Experiment A — Isaac only

Dataset:

```text
../data/refined_isaac_0908
```

## A1. Cheap — 15% train sequences, 8 epochs

```bash
uv run python scripts/train.py \
  --config configs/cheap.yaml \
  --dataset-root "$ISAAC_ROOT" \
  --run-dir "$RUNS_ROOT/isaac_cheap_${DATE}" \
  --device cuda
```

## A2. Baseline — 50% train sequences, 30 epochs

```bash
uv run python scripts/train.py \
  --config configs/baseline.yaml \
  --dataset-root "$ISAAC_ROOT" \
  --run-dir "$RUNS_ROOT/isaac_baseline_${DATE}" \
  --device cuda
```

## A3. Optuna — same 50% as baseline, 20 trials × 15 epochs

```bash
uv run python scripts/tune_optuna.py \
  --config configs/optuna.yaml \
  --dataset-root "$ISAAC_ROOT" \
  --run-dir "$RUNS_ROOT/isaac_optuna_${DATE}" \
  --device cuda \
  --n-trials 20
```

Best tuned parameters are written to:

```text
../runs_nn/isaac_optuna_0908/best_overrides.yaml
```

## A4. Final — 100% train sequences, 60 epochs

This applies the Isaac-only Optuna result to the full Isaac training split:

```bash
uv run python scripts/train.py \
  --config configs/final.yaml \
  --overrides "$RUNS_ROOT/isaac_optuna_${DATE}/best_overrides.yaml" \
  --dataset-root "$ISAAC_ROOT" \
  --run-dir "$RUNS_ROOT/isaac_final_${DATE}" \
  --device cuda
```

## A5. Final Isaac test evaluation

Only run this after model selection/final training:

```bash
uv run python scripts/evaluate_checkpoint.py \
  --checkpoint "$RUNS_ROOT/isaac_final_${DATE}/best.pt" \
  --dataset-root "$ISAAC_ROOT" \
  --split test \
  --device cuda \
  --output "$RUNS_ROOT/isaac_final_${DATE}/evaluation_test.json" \
  --per-sample-output "$RUNS_ROOT/isaac_final_${DATE}/per_sample_metrics_test.csv"
```

# Experiment B — Synthetic + Isaac combined

Datasets:

```text
../data/synth_large_0906
+
../data/refined_isaac_0908
```

To combine datasets, repeat `--dataset-root`. No copied/merged data folder is needed.

For fractional training, the same percentage is deterministically selected from **each source's train sequence manifest**, preserving the source mixture:

- cheap: 15% synthetic train sequences + 15% Isaac train sequences,
- baseline/Optuna: 50% synthetic + 50% Isaac,
- final: 100% synthetic + 100% Isaac.

## B1. Cheap — 15% per source, 8 epochs

```bash
uv run python scripts/train.py \
  --config configs/cheap.yaml \
  --dataset-root "$SYNTH_ROOT" \
  --dataset-root "$ISAAC_ROOT" \
  --run-dir "$RUNS_ROOT/combined_cheap_${DATE}" \
  --device cuda
```

## B2. Baseline — 50% per source, 30 epochs

```bash
uv run python scripts/train.py \
  --config configs/baseline.yaml \
  --dataset-root "$SYNTH_ROOT" \
  --dataset-root "$ISAAC_ROOT" \
  --run-dir "$RUNS_ROOT/combined_baseline_${DATE}" \
  --device cuda
```

## B3. Optuna — exact same 50% as combined baseline, 20 trials × 15 epochs

```bash
uv run python scripts/tune_optuna.py \
  --config configs/optuna.yaml \
  --dataset-root "$SYNTH_ROOT" \
  --dataset-root "$ISAAC_ROOT" \
  --run-dir "$RUNS_ROOT/combined_optuna_${DATE}" \
  --device cuda \
  --n-trials 20
```

Best tuned parameters are written to:

```text
../runs_nn/combined_optuna_0908/best_overrides.yaml
```

## B4. Final — 100% of both sources, 60 epochs

```bash
uv run python scripts/train.py \
  --config configs/final.yaml \
  --overrides "$RUNS_ROOT/combined_optuna_${DATE}/best_overrides.yaml" \
  --dataset-root "$SYNTH_ROOT" \
  --dataset-root "$ISAAC_ROOT" \
  --run-dir "$RUNS_ROOT/combined_final_${DATE}" \
  --device cuda
```

## B5. Final combined test evaluation

This evaluates on the fixed test manifests from **both** sources:

```bash
uv run python scripts/evaluate_checkpoint.py \
  --checkpoint "$RUNS_ROOT/combined_final_${DATE}/best.pt" \
  --dataset-root "$SYNTH_ROOT" \
  --dataset-root "$ISAAC_ROOT" \
  --split test \
  --device cuda \
  --output "$RUNS_ROOT/combined_final_${DATE}/evaluation_test.json" \
  --per-sample-output "$RUNS_ROOT/combined_final_${DATE}/per_sample_metrics_test.csv"
```

# Video inference / annotated MP4 outputs

`infer_video.py` is a direct **input video -> annotated output video** command while still preserving the canonical machine-readable prediction outputs.

For every detected gate it draws:

- a different deterministic translucent color per persistent `track_id` (or `mask_id` when tracking is disabled),
- the predicted instance-mask overlay,
- a thin instance contour,
- a bounding box,
- predicted keypoints unless `--no-keypoints` is supplied,
- a compact instance label with track ID, detection confidence, and predicted optical depth.

The top-left HUD contains small text showing:

- frame index,
- number of detected gates,
- mean detection confidence,
- network inference time,
- postprocessing time,
- total model+postprocess latency,
- effective inference FPS,
- source-video FPS.

Predicted masks, boxes, and keypoints are mapped back to the **original source-video resolution** before they are written or drawn.

Set the roots and the date used by the training checkpoints. If the models were trained on September 8, for example:

```bash
export RUNS_ROOT="../runs_nn"
export OUTPUTS_ROOT="../outputs_nn"
export RUN_DATE="0908"       # date suffix on the checkpoint folders
export OUT_DATE="$(date +%m%d)"  # date suffix for this inference output
export VIDEO="/path/to/input_video.mp4"

mkdir -p "$OUTPUTS_ROOT"
```

If you are running inference the same day the checkpoints were trained, you can simply use:

```bash
export RUN_DATE="$(date +%m%d)"
export OUT_DATE="$RUN_DATE"
```

## Isaac-only checkpoints

### Isaac cheap

```bash
uv run python scripts/infer_video.py \
  --checkpoint "$RUNS_ROOT/isaac_cheap_${RUN_DATE}/best.pt" \
  --video "$VIDEO" \
  --output "$OUTPUTS_ROOT/isaac_cheap_${OUT_DATE}" \
  --overlay-video "isaac_cheap_overlay_${OUT_DATE}.mp4" \
  --device cuda
```

### Isaac baseline

```bash
uv run python scripts/infer_video.py \
  --checkpoint "$RUNS_ROOT/isaac_baseline_${RUN_DATE}/best.pt" \
  --video "$VIDEO" \
  --output "$OUTPUTS_ROOT/isaac_baseline_${OUT_DATE}" \
  --overlay-video "isaac_baseline_overlay_${OUT_DATE}.mp4" \
  --device cuda
```

### Isaac Optuna best trial

Resolve the best trial checkpoint automatically from `study_summary.yaml`:

```bash
export ISAAC_OPTUNA_BEST="$(uv run python scripts/resolve_optuna_best_checkpoint.py \
  --run-dir "$RUNS_ROOT/isaac_optuna_${RUN_DATE}")"

echo "$ISAAC_OPTUNA_BEST"
```

Then infer:

```bash
uv run python scripts/infer_video.py \
  --checkpoint "$ISAAC_OPTUNA_BEST" \
  --video "$VIDEO" \
  --output "$OUTPUTS_ROOT/isaac_optuna_best_${OUT_DATE}" \
  --overlay-video "isaac_optuna_best_overlay_${OUT_DATE}.mp4" \
  --device cuda
```

### Isaac final

```bash
uv run python scripts/infer_video.py \
  --checkpoint "$RUNS_ROOT/isaac_final_${RUN_DATE}/best.pt" \
  --video "$VIDEO" \
  --output "$OUTPUTS_ROOT/isaac_final_${OUT_DATE}" \
  --overlay-video "isaac_final_overlay_${OUT_DATE}.mp4" \
  --device cuda
```

## Synthetic + Isaac combined checkpoints

### Combined cheap

```bash
uv run python scripts/infer_video.py \
  --checkpoint "$RUNS_ROOT/combined_cheap_${RUN_DATE}/best.pt" \
  --video "$VIDEO" \
  --output "$OUTPUTS_ROOT/combined_cheap_${OUT_DATE}" \
  --overlay-video "combined_cheap_overlay_${OUT_DATE}.mp4" \
  --device cuda
```

### Combined baseline

```bash
uv run python scripts/infer_video.py \
  --checkpoint "$RUNS_ROOT/combined_baseline_${RUN_DATE}/best.pt" \
  --video "$VIDEO" \
  --output "$OUTPUTS_ROOT/combined_baseline_${OUT_DATE}" \
  --overlay-video "combined_baseline_overlay_${OUT_DATE}.mp4" \
  --device cuda
```

### Combined Optuna best trial

```bash
export COMBINED_OPTUNA_BEST="$(uv run python scripts/resolve_optuna_best_checkpoint.py \
  --run-dir "$RUNS_ROOT/combined_optuna_${RUN_DATE}")"

echo "$COMBINED_OPTUNA_BEST"
```

Then infer:

```bash
uv run python scripts/infer_video.py \
  --checkpoint "$COMBINED_OPTUNA_BEST" \
  --video "$VIDEO" \
  --output "$OUTPUTS_ROOT/combined_optuna_best_${OUT_DATE}" \
  --overlay-video "combined_optuna_best_overlay_${OUT_DATE}.mp4" \
  --device cuda
```

### Combined final

```bash
uv run python scripts/infer_video.py \
  --checkpoint "$RUNS_ROOT/combined_final_${RUN_DATE}/best.pt" \
  --video "$VIDEO" \
  --output "$OUTPUTS_ROOT/combined_final_${OUT_DATE}" \
  --overlay-video "combined_final_overlay_${OUT_DATE}.mp4" \
  --device cuda
```

Each output directory now contains:

```text
../outputs_nn/<name>_MMDD/
├── inference.json
├── frames.jsonl
├── <name>_overlay_MMDD.mp4
└── instance_masks/
    ├── frame_000000.png
    ├── frame_000001.png
    └── ...
```

The integer mask encoding remains:

```text
0 = background
1 = predicted gate instance 1
2 = predicted gate instance 2
...
```

`frames.jsonl` maps each frame-local `mask_id` to the predicted `track_id`, confidence, box, keypoints/visibility, pose, and runtime metrics.

### Useful inference options

Make masks more or less transparent:

```bash
--overlay-alpha 0.30
```

Hide keypoint dots while keeping masks/boxes/text:

```bash
--no-keypoints
```

Disable temporal tracking:

```bash
--no-tracking
```

When tracking is enabled, each persistent `track_id` receives a stable overlay color across frames. When tracking is disabled, colors are assigned from frame-local `mask_id`s.

# Exact deterministic data-selection behavior

All configs contain:

```yaml
seed: 42
deterministic: true
```

and the budgets are:

| Stage | Train sequences | Validation | Epochs / trials |
|---|---:|---:|---:|
| Cheap | 15% | 100% | 8 epochs |
| Baseline | 50% | 100% | 30 epochs |
| Optuna | same 50% | 100% | 20 trials × 15 epochs |
| Final | 100% | 100% | 60 epochs |

The split files themselves remain authoritative:

```text
splits/train_sequences.txt
splits/validation_sequences.txt
splits/test_sequences.txt
```

Fractional selection occurs **only inside `train_sequences.txt`**. It never moves sequences between train/validation/test.

Within each source, train sequences are assigned a deterministic SHA-256 rank from:

```text
seed 42 + source dataset name + sequence ID
```

The first 15% are cheap; the first 50% are baseline/Optuna. Therefore:

```text
cheap 15% ⊂ baseline 50% = Optuna 50% ⊂ final 100%
```

Every training run writes:

```text
data_selection.json
```

containing the exact sequence IDs used. This lets you verify that two model runs truly trained on the same trajectories.

PyTorch, CUDA, Python `random`, NumPy, DataLoader shuffling/workers, augmentations, and the Optuna TPE sampler are all seeded from 42. Deterministic PyTorch algorithms are enabled with `warn_only=True` where supported.

# Files created by each training run

```text
../runs_nn/<run_name>_MMDD/
├── resolved_config.yaml
├── data_selection.json
├── best.pt
├── last.pt
├── training_history.csv
├── per_sample_metrics.csv
└── evaluation.json
```

Optuna additionally creates:

```text
../runs_nn/<name>_optuna_MMDD/
├── trial_0000/
├── trial_0001/
├── ...
├── best_overrides.yaml
└── study_summary.yaml
```

# Important comparison rule

For meaningful model comparisons, use the same:

- canonical dataset roots,
- fixed train/validation/test manifests,
- seed 42,
- stage (`cheap`, `baseline`, etc.),
- sequence fraction,
- input resolution/window,
- evaluation split.

The cheap, baseline, and final configs intentionally use the **same underlying GatePerceiver architecture**. Their primary differences are training-data fraction and epoch budget; Optuna tunes hyperparameters on the baseline 50% subset before the tuned values are used for final full-data training.

# Apples-to-apples comparison on the Isaac test set

For comparing the effect of adding synthetic training data, both final models should be scored on the **same Isaac-only test set**.

The Isaac-only final command above already does this. Evaluate the combined-trained model on that same Isaac test manifest with:

```bash
uv run python scripts/evaluate_checkpoint.py \
  --checkpoint "$RUNS_ROOT/combined_final_${DATE}/best.pt" \
  --dataset-root "$ISAAC_ROOT" \
  --split test \
  --device cuda \
  --output "$RUNS_ROOT/combined_final_${DATE}/evaluation_test_isaac_only.json" \
  --per-sample-output "$RUNS_ROOT/combined_final_${DATE}/per_sample_metrics_test_isaac_only.csv"
```

The cleanest final comparison is therefore:

```text
Isaac-trained model       -> Isaac test set
Combined-trained model    -> exact same Isaac test set
```

You may additionally retain the combined-source test evaluation to understand performance across both domains, but use the Isaac-only holdout for the primary deployment comparison.
