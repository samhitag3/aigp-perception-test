# Union MaskPoseNet + training-only realistic mask noise

This version keeps the union-mask MaskPoseNet architecture unchanged and adds
**training-only input corruption**. The original canonical instance masks remain
the source of truth for every loss and every validation/test metric.

## What is corrupted during training

For each temporal window, one correlated noise profile is sampled. Individual
frames vary mildly inside that profile. Noise includes:

- rounded/softened mask edges;
- erosion/dilation and a few-pixel boundary/registration shift;
- missing chunks / missing gate bars;
- rare complete gate misses;
- very rare whole-frame mask dropout so the ConvGRU learns to bridge gaps;
- persistent partial square/ring false positives;
- arbitrary false polygons;
- small false blobs.

A fraction of windows remains completely clean. Validation, test, presence
threshold calibration and reported accuracy always use the original clean masks.

## Setup

```bash
cd maskposenet_union_noise
uv sync
source .venv/bin/activate

DATE=0916
SYNTH=../data/synth_large_0906
ISAAC=../data/refined_isaac_0908
RUNS=../runs_union_noise
OUTS=../outputs_union_noise
CFG=configs/maskposenet/combined.yaml
mkdir -p "$RUNS" "$OUTS"
```

## Prepare deterministic subsets

```bash
uv run python scripts/prepare_vanilla_splits.py \
  --dataset-root "$SYNTH" \
  --seed 42 \
  --cheap-fraction 0.15 \
  --baseline-fraction 0.50

uv run python scripts/prepare_vanilla_splits.py \
  --dataset-root "$ISAAC" \
  --seed 42 \
  --cheap-fraction 0.15 \
  --baseline-fraction 0.50
```

## Validate original datasets

```bash
uv run python scripts/validate_mask_dataset.py \
  --dataset-root "$SYNTH" \
  --width 640 \
  --height 360

uv run python scripts/validate_mask_dataset.py \
  --dataset-root "$ISAAC" \
  --width 640 \
  --height 360
```

## Preview the training corruption

This does not modify the dataset.

```bash
uv run python scripts/preview_mask_noise.py \
  --config "$CFG" \
  --dataset-root "$ISAAC" \
  --sequence-id env_000_episode_0000001 \
  --out-dir "$OUTS/noise_preview_${DATE}" \
  --frames 24 \
  --seed 42 \
  --force-noisy
```

Outputs include clean/noisy PNGs, side-by-side comparisons, `noise_preview.mp4`
and `preview.json`.

## Cheap — 15%, 8 epochs

```bash
uv run python scripts/train_maskposenet.py \
  --config "$CFG" \
  --train-manifest splits/vanilla_seed42/train_cheap_sequences.txt \
  --epochs 8 \
  --run-dir "$RUNS/cheap_${DATE}" \
  --device cuda

uv run python scripts/evaluate_maskposenet.py \
  --config "$CFG" \
  --checkpoint "$RUNS/cheap_${DATE}/best.pt" \
  --split validation \
  --out-dir "$OUTS/cheap_val_${DATE}" \
  --device cuda

uv run python scripts/benchmark_maskposenet.py \
  --config "$CFG" \
  --checkpoint "$RUNS/cheap_${DATE}/best.pt" \
  --device cuda \
  --out "$OUTS/cheap_latency_${DATE}.json"
```

## Cheap inference

Inference does **not** add artificial noise. Give it whatever masks your real
upstream segmenter actually produces.

```bash
uv run python scripts/infer_maskposenet.py \
  --input "$ISAAC/sequences/env_000_episode_0000001/instance_masks" \
  --checkpoint "$RUNS/cheap_${DATE}/best.pt" \
  --config "$CFG" \
  --out-dir "$OUTS/cheap_infer_${DATE}" \
  --fps 60 \
  --device cuda
```

Optional RGB visualization only:

```bash
uv run python scripts/infer_maskposenet.py \
  --input "$ISAAC/sequences/env_000_episode_0000001/instance_masks" \
  --rgb-input "$ISAAC/sequences/env_000_episode_0000001/rgb" \
  --checkpoint "$RUNS/cheap_${DATE}/best.pt" \
  --config "$CFG" \
  --out-dir "$OUTS/cheap_infer_rgb_${DATE}" \
  --fps 60 \
  --device cuda
```

## Baseline — 50%, 30 epochs

```bash
uv run python scripts/train_maskposenet.py \
  --config "$CFG" \
  --train-manifest splits/vanilla_seed42/train_baseline_sequences.txt \
  --epochs 30 \
  --run-dir "$RUNS/baseline_${DATE}" \
  --device cuda

uv run python scripts/evaluate_maskposenet.py \
  --config "$CFG" \
  --checkpoint "$RUNS/baseline_${DATE}/best.pt" \
  --split validation \
  --out-dir "$OUTS/baseline_val_${DATE}" \
  --device cuda

uv run python scripts/benchmark_maskposenet.py \
  --config "$CFG" \
  --checkpoint "$RUNS/baseline_${DATE}/best.pt" \
  --device cuda \
  --out "$OUTS/baseline_latency_${DATE}.json"
```

Baseline inference with RGB visualization:

```bash
uv run python scripts/infer_maskposenet.py \
  --input "$ISAAC/sequences/env_000_episode_0000001/instance_masks" \
  --rgb-input "$ISAAC/sequences/env_000_episode_0000001/rgb" \
  --checkpoint "$RUNS/baseline_${DATE}/best.pt" \
  --config "$CFG" \
  --out-dir "$OUTS/baseline_infer_rgb_${DATE}" \
  --fps 60 \
  --device cuda
```

## Model Optuna — same 50%, 20 trials x 15 epochs

The temporal window remains searchable over `[4, 6, 8, 10, 12, 16]`. Noise
parameters are intentionally fixed rather than tuned against clean validation;
otherwise Optuna can favor unrealistically weak noise simply to improve the
clean objective.

```bash
uv run python scripts/tune_maskposenet_optuna.py \
  --config "$CFG" \
  --train-manifest splits/vanilla_seed42/train_tune_sequences.txt \
  --val-manifest splits/validation_sequences.txt \
  --run-dir "$RUNS/optuna_${DATE}" \
  --output-dir "$OUTS/optuna_${DATE}" \
  --trials 20 \
  --epochs-per-trial 15 \
  --device cuda

BEST_CFG="$OUTS/optuna_${DATE}/best_config.yaml"
```

## Final — 100%, 60 epochs

```bash
BEST_CFG="$OUTS/optuna_${DATE}/best_config.yaml"

uv run python scripts/train_maskposenet.py \
  --config "$BEST_CFG" \
  --train-manifest splits/train_sequences.txt \
  --epochs 60 \
  --run-dir "$RUNS/final_${DATE}" \
  --device cuda
```

## Presence-threshold tuning — final checkpoint, clean validation

```bash
uv run python scripts/tune_presence_threshold_optuna.py \
  --config "$BEST_CFG" \
  --checkpoint "$RUNS/final_${DATE}/best.pt" \
  --split validation \
  --out-dir "$OUTS/presence_tune_${DATE}" \
  --trials 100 \
  --min-th 0.05 \
  --max-th 0.95 \
  --objective far_balanced \
  --device cuda

FINAL_CFG="$OUTS/presence_tune_${DATE}/best_config_with_threshold.yaml"
```

## Final clean test

```bash
uv run python scripts/evaluate_maskposenet.py \
  --config "$FINAL_CFG" \
  --checkpoint "$RUNS/final_${DATE}/best.pt" \
  --split test \
  --out-dir "$OUTS/final_test_${DATE}" \
  --device cuda
```

## Final latency

```bash
uv run python scripts/benchmark_maskposenet.py \
  --config "$FINAL_CFG" \
  --checkpoint "$RUNS/final_${DATE}/best.pt" \
  --device cuda \
  --out "$OUTS/final_latency_${DATE}.json"
```

## Final inference

Mask-only visualization:

```bash
uv run python scripts/infer_maskposenet.py \
  --input "$ISAAC/sequences/env_000_episode_0000001/instance_masks" \
  --checkpoint "$RUNS/final_${DATE}/best.pt" \
  --config "$FINAL_CFG" \
  --out-dir "$OUTS/final_infer_${DATE}" \
  --fps 60 \
  --device cuda
```

RGB background for visualization only:

```bash
uv run python scripts/infer_maskposenet.py \
  --input "$ISAAC/sequences/env_000_episode_0000001/instance_masks" \
  --rgb-input "$ISAAC/sequences/env_000_episode_0000001/rgb" \
  --checkpoint "$RUNS/final_${DATE}/best.pt" \
  --config "$FINAL_CFG" \
  --out-dir "$OUTS/final_infer_rgb_${DATE}" \
  --fps 60 \
  --device cuda
```
