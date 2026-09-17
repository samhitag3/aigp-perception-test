# Union MaskPoseNet — 0915 workflow

This fork makes one data-input change relative to MaskPoseNet: **all non-zero input mask IDs are unionized to one foreground value before entering the model**. The network architecture, ConvGRU, 8-query decoder, per-gate output masks, keypoints, pose, target head, losses, latency benchmark, Optuna model tuning, and presence-threshold calibration remain unchanged.

Important: unionization applies only to the model input tensor. The original integer instance-mask image is retained inside the dataset loader for per-gate GT output supervision.

## Setup / paths

```bash
uv sync
source .venv/bin/activate

DATE=0915
SYNTH=../data/synth_large_0906
ISAAC=../data/refined_isaac_0908
RUNS=../runs_union
OUTS=../outputs_union
CFG=configs/maskposenet/combined.yaml
mkdir -p "$RUNS" "$OUTS"
```

## Prepare deterministic subsets

```bash
uv run python scripts/prepare_vanilla_splits.py --dataset-root "$SYNTH" --seed 42 --cheap-fraction 0.15 --baseline-fraction 0.50
uv run python scripts/prepare_vanilla_splits.py --dataset-root "$ISAAC" --seed 42 --cheap-fraction 0.15 --baseline-fraction 0.50
```

## Validate both datasets

```bash
uv run python scripts/validate_mask_dataset.py --dataset-root "$SYNTH" --width 640 --height 360
uv run python scripts/validate_mask_dataset.py --dataset-root "$ISAAC" --width 640 --height 360
```

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

## Model Optuna — same 50% subset, 20 x 15 epochs

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
```

Best model config:

```bash
BEST_CFG="$OUTS/optuna_${DATE}/best_config.yaml"
```

## Final model — 100%, 60 epochs

```bash
BEST_CFG="$OUTS/optuna_${DATE}/best_config.yaml"

uv run python scripts/train_maskposenet.py \
  --config "$BEST_CFG" \
  --train-manifest splits/train_sequences.txt \
  --epochs 60 \
  --run-dir "$RUNS/final_${DATE}" \
  --device cuda
```

## Presence-threshold Optuna — FINAL checkpoint, validation only

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

## Final test + benchmark

```bash
uv run python scripts/evaluate_maskposenet.py \
  --config "$FINAL_CFG" \
  --checkpoint "$RUNS/final_${DATE}/best.pt" \
  --split test \
  --out-dir "$OUTS/final_test_${DATE}" \
  --device cuda

uv run python scripts/benchmark_maskposenet.py \
  --config "$FINAL_CFG" \
  --checkpoint "$RUNS/final_${DATE}/best.pt" \
  --device cuda \
  --out "$OUTS/final_latency_${DATE}.json"
```

## Inference — union/binary mask input

The input may be a binary mask sequence (`0=background`, `1=gate`) or the old instance-ID masks (`0,1,2,...`). In both cases every non-zero pixel is unionized before entering the model.

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

Optional RGB visualization only:

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
