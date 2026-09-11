# GatePoseNet-MG Canonical Training Guide

This keeps the existing GatePoseNet-MG architecture unchanged and makes its loader/training pipeline consume the canonical UAV gate perception dataset contract.

## What MG learns

For each frame/window, GatePoseNet-MG receives RGB + ego motion and predicts up to 8 gate queries. Each query predicts:

- presence score
- target-gate score
- one per-gate mask
- four outer corners
- center
- camera-relative 3-D position
- log-depth
- 6-D rotation
- visible fraction

The canonical integer instance PNG is converted per gate using `instance_mask == mask_id`. It is NOT unionized.

## Dataset regimes

1. Isaac only: `../data/refined_isaac_0908`
2. Synthetic + Isaac: `../data/synth_large_0906` + `../data/refined_isaac_0908`

Use the same deterministic sequence manifests as the vanilla baseline so comparisons are apples-to-apples:

- cheap: 15% train sequences, seed 42
- baseline: 50% train sequences, seed 42
- Optuna: exact same 50% as baseline, 20 trials x 15 epochs
- final: 100% official train split, 60 epochs

Validation/test use complete fixed sequence-level splits.

## Setup

```bash
uv sync
uv run python scripts/check_gpu.py

STAMP=$(date +%m%d)
RUNS=../runs_vanilla
OUTS=../outputs_vanilla
MG_ISAAC=configs/mg/isaac.yaml
MG_COMBINED=configs/mg/combined.yaml
mkdir -p "$RUNS" "$OUTS"
```

## Validate / prepare deterministic manifests

If the `splits/vanilla_seed42/` manifests already exist from the vanilla experiments, reuse them and skip the preparation commands.

```bash
uv run python scripts/validate_dataset.py --dataset-root ../data/refined_isaac_0908
uv run python scripts/validate_dataset.py --dataset-root ../data/synth_large_0906
```

If needed:

```bash
uv run python scripts/prepare_vanilla_splits.py \
  --dataset-root ../data/refined_isaac_0908 \
  --seed 42 --cheap-fraction 0.15 --baseline-fraction 0.50 \
  --create-base-if-missing

uv run python scripts/prepare_vanilla_splits.py \
  --dataset-root ../data/synth_large_0906 \
  --seed 42 --cheap-fraction 0.15 --baseline-fraction 0.50 \
  --create-base-if-missing
```

# A. Isaac only

## Cheap — 15%, 8 epochs

```bash
uv run python scripts/train_gatepose_mg_contract.py \
  --config "$MG_ISAAC" \
  --train-manifest splits/vanilla_seed42/train_cheap_sequences.txt \
  --epochs 8 \
  --run-dir "$RUNS/mg_isaac_cheap_${STAMP}" \
  --device cuda
```

Validation:

```bash
uv run python scripts/evaluate_gatepose_mg_contract.py \
  --config "$MG_ISAAC" \
  --checkpoint "$RUNS/mg_isaac_cheap_${STAMP}/best.pt" \
  --split validation \
  --out-dir "$OUTS/mg_isaac_cheap_val_${STAMP}" \
  --device cuda
```

## Baseline — 50%, 30 epochs

```bash
uv run python scripts/train_gatepose_mg_contract.py \
  --config "$MG_ISAAC" \
  --train-manifest splits/vanilla_seed42/train_baseline_sequences.txt \
  --epochs 30 \
  --run-dir "$RUNS/mg_isaac_baseline_${STAMP}" \
  --device cuda
```

Validation:

```bash
uv run python scripts/evaluate_gatepose_mg_contract.py \
  --config "$MG_ISAAC" \
  --checkpoint "$RUNS/mg_isaac_baseline_${STAMP}/best.pt" \
  --split validation \
  --out-dir "$OUTS/mg_isaac_baseline_val_${STAMP}" \
  --device cuda
```

## Optuna — same exact 50%, 20 x 15 epochs

```bash
uv run python scripts/tune_gatepose_mg_optuna.py \
  --config "$MG_ISAAC" \
  --train-manifest splits/vanilla_seed42/train_tune_sequences.txt \
  --val-manifest splits/validation_sequences.txt \
  --run-dir "$RUNS/mg_isaac_optuna_${STAMP}" \
  --output-dir "$OUTS/mg_isaac_optuna_${STAMP}" \
  --trials 20 \
  --epochs-per-trial 15 \
  --device cuda
```

## Final — 100%, tuned config, 60 epochs

```bash
uv run python scripts/train_gatepose_mg_contract.py \
  --config "$OUTS/mg_isaac_optuna_${STAMP}/best_config.yaml" \
  --train-manifest splits/train_sequences.txt \
  --epochs 60 \
  --run-dir "$RUNS/mg_isaac_final_${STAMP}" \
  --device cuda
```

Final test:

```bash
uv run python scripts/evaluate_gatepose_mg_contract.py \
  --config "$OUTS/mg_isaac_optuna_${STAMP}/best_config.yaml" \
  --checkpoint "$RUNS/mg_isaac_final_${STAMP}/best.pt" \
  --split test \
  --out-dir "$OUTS/mg_isaac_final_test_${STAMP}" \
  --device cuda
```

# B. Synthetic + Isaac

The same manifest filename is resolved independently under both canonical dataset roots.

## Cheap — 15% of each source, 8 epochs

```bash
uv run python scripts/train_gatepose_mg_contract.py \
  --config "$MG_COMBINED" \
  --train-manifest splits/vanilla_seed42/train_cheap_sequences.txt \
  --epochs 8 \
  --run-dir "$RUNS/mg_synth_isaac_cheap_${STAMP}" \
  --device cuda
```

Validation:

```bash
uv run python scripts/evaluate_gatepose_mg_contract.py \
  --config "$MG_COMBINED" \
  --checkpoint "$RUNS/mg_synth_isaac_cheap_${STAMP}/best.pt" \
  --split validation \
  --out-dir "$OUTS/mg_synth_isaac_cheap_val_${STAMP}" \
  --device cuda
```

## Baseline — 50% of each source, 30 epochs

```bash
uv run python scripts/train_gatepose_mg_contract.py \
  --config "$MG_COMBINED" \
  --train-manifest splits/vanilla_seed42/train_baseline_sequences.txt \
  --epochs 30 \
  --run-dir "$RUNS/mg_synth_isaac_baseline_${STAMP}" \
  --device cuda
```

Validation:

```bash
uv run python scripts/evaluate_gatepose_mg_contract.py \
  --config "$MG_COMBINED" \
  --checkpoint "$RUNS/mg_synth_isaac_baseline_${STAMP}/best.pt" \
  --split validation \
  --out-dir "$OUTS/mg_synth_isaac_baseline_val_${STAMP}" \
  --device cuda
```

## Optuna — exact same 50%, 20 x 15 epochs

```bash
uv run python scripts/tune_gatepose_mg_optuna.py \
  --config "$MG_COMBINED" \
  --train-manifest splits/vanilla_seed42/train_tune_sequences.txt \
  --val-manifest splits/validation_sequences.txt \
  --run-dir "$RUNS/mg_synth_isaac_optuna_${STAMP}" \
  --output-dir "$OUTS/mg_synth_isaac_optuna_${STAMP}" \
  --trials 20 \
  --epochs-per-trial 15 \
  --device cuda
```

## Final — 100% both source train splits, tuned config, 60 epochs

```bash
uv run python scripts/train_gatepose_mg_contract.py \
  --config "$OUTS/mg_synth_isaac_optuna_${STAMP}/best_config.yaml" \
  --train-manifest splits/train_sequences.txt \
  --epochs 60 \
  --run-dir "$RUNS/mg_synth_isaac_final_${STAMP}" \
  --device cuda
```

Final test:

```bash
uv run python scripts/evaluate_gatepose_mg_contract.py \
  --config "$OUTS/mg_synth_isaac_optuna_${STAMP}/best_config.yaml" \
  --checkpoint "$RUNS/mg_synth_isaac_final_${STAMP}/best.pt" \
  --split test \
  --out-dir "$OUTS/mg_synth_isaac_final_test_${STAMP}" \
  --device cuda
```

## Key compatibility notes

- RGB input remains 3-channel RGB.
- Canonical 640x360 intrinsics remain `fx=fy=320, cx=320, cy=180` (HFoV 90 deg; VFoV about 58.716 deg).
- Network training resolution remains the original 320x192; 2-D coordinates are normalized before resize, so pose labels remain in metric camera coordinates.
- `n_queries=8`, `max_gates=8`. If more than 8 valid gates exist in a frame, the adapter deterministically prioritizes the current target, then visible/route-relevant/nearer gates.
- The MG architecture predicts four OUTER corners per gate, not all eight outer+inner corners.
- The target head still identifies which predicted gate is the current route target; unlike GatePoseNetSingle, the other gates are also predicted independently.
- Checkpoint selection uses median visible target-gate position error on validation, matching the original MG training script.
