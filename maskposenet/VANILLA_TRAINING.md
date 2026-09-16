# Vanilla GatePoseNet — Isaac-only and Synthetic+Isaac training

This guide is the canonical command sequence for the unchanged original
`GatePoseNetSingle` running through the canonical dataset compatibility layer.

## Data regimes

Two deliberately separate experiments are supported:

1. **Isaac-only**: `../data/refined_isaac_0908`
2. **Synthetic+Isaac**: `../data/synth_large_0906` + `../data/refined_isaac_0908`

The combined regime concatenates selected sequences from the two source datasets.
There is **no source oversampling or source-specific weighting** in this vanilla
comparison.

## Fixed experiment funnel

All selection happens at the **sequence** level. Frames from one sequence never
cross train/validation/test boundaries.

| Stage | Train data | Epochs | Purpose |
|---|---:|---:|---|
| Cheap | deterministic 15% of official train sequences | 8 | fast architecture smoke/baseline |
| Baseline | deterministic 50% of official train sequences | 30 | meaningful default-settings baseline |
| Optuna | **same exact 50% sequences as baseline** | 15/trial, 20 trials | tune optimizer/loss settings |
| Final | 100% of official train sequences | 60 | full tuned training |

The cheap subset is nested inside the baseline subset. If a dataset already has
`splits/train_cheap_sequences.txt` (for example from an earlier seed-42 run),
that exact cheap manifest is reused so old and new model results stay directly
comparable. Otherwise cheap is generated with seed 42 and a stable SHA-256
ordering. Baseline is then deterministically completed around that cheap subset.
Validation and test manifests remain full-size.
The test split is used **only after final training**.

Optuna keeps the GatePoseNet architecture unchanged. It tunes only training/loss
settings (`lr`, weight decay, gradient clipping, ego dropout, and selected loss
weights). Every trial uses training seed 42; Optuna's TPE sampler also uses seed
42.

---

## 0. One-time setup

From the repository root:

```bash
uv sync
uv run python scripts/check_gpu.py

STAMP=$(date +%m%d)       # Sep 8 -> 0908
RUNS=../runs_vanilla
OUTS=../outputs_vanilla
ISAAC_CFG=configs/vanilla/isaac.yaml
COMBINED_CFG=configs/vanilla/combined.yaml

mkdir -p "$RUNS" "$OUTS"
```

Camera convention in both configs:

```text
native image = 640 x 360
fx = fy = 320
cx = 320
cy = 180
HFoV = 90.000 deg
VFoV = 58.716 deg
```

The historical `90 deg VFoV` label is treated as a mislabeled HFoV.

---

## 1. Prepare and validate the fixed sequence splits

Run this once for **each source dataset**. Existing official train/validation/test
manifests are preserved. If they are absent, `--create-base-if-missing` creates a
deterministic 80/10/10 sequence split.

The script also normalizes the older `splits/val_sequences.txt` alias to the
canonical `splits/validation_sequences.txt` without changing its contents.

### Isaac

```bash
uv run python scripts/prepare_vanilla_splits.py \
  --dataset-root ../data/refined_isaac_0908 \
  --seed 42 \
  --cheap-fraction 0.15 \
  --baseline-fraction 0.50 \
  --create-base-if-missing

uv run python scripts/validate_dataset.py \
  --dataset-root ../data/refined_isaac_0908
```

### Synthetic

```bash
uv run python scripts/prepare_vanilla_splits.py \
  --dataset-root ../data/synth_large_0906 \
  --seed 42 \
  --cheap-fraction 0.15 \
  --baseline-fraction 0.50 \
  --create-base-if-missing

uv run python scripts/validate_dataset.py \
  --dataset-root ../data/synth_large_0906
```

Each dataset will then contain:

```text
splits/
├── train_sequences.txt
├── validation_sequences.txt
├── test_sequences.txt
└── vanilla_seed42/
    ├── train_cheap_sequences.txt       # 15% of train
    ├── train_baseline_sequences.txt    # 50% of train
    ├── train_tune_sequences.txt        # exact copy of baseline IDs
    └── manifest_info.json
```

Do not regenerate these with another seed when comparing models.

---

# EXPERIMENT A — ISAAC ONLY

## A1. Cheap training — 15%, 8 epochs

```bash
uv run python scripts/train_gatepose.py \
  --config "$ISAAC_CFG" \
  --train-manifest splits/vanilla_seed42/train_cheap_sequences.txt \
  --epochs 8 \
  --run-dir "$RUNS/isaac_cheap_${STAMP}" \
  --device cuda \
  --skip-final-eval
```

Evaluate on the **full validation split**:

```bash
uv run python scripts/evaluate_gatepose_contract.py \
  --config "$ISAAC_CFG" \
  --checkpoint "$RUNS/isaac_cheap_${STAMP}/best.pt" \
  --split validation \
  --out-dir "$OUTS/isaac_cheap_val_${STAMP}" \
  --device cuda
```

Outputs:

```text
../runs_vanilla/isaac_cheap_0908/
../outputs_vanilla/isaac_cheap_val_0908/
```

## A2. Baseline training — 50%, 30 epochs

```bash
uv run python scripts/train_gatepose.py \
  --config "$ISAAC_CFG" \
  --train-manifest splits/vanilla_seed42/train_baseline_sequences.txt \
  --epochs 30 \
  --run-dir "$RUNS/isaac_baseline_${STAMP}" \
  --device cuda \
  --skip-final-eval
```

```bash
uv run python scripts/evaluate_gatepose_contract.py \
  --config "$ISAAC_CFG" \
  --checkpoint "$RUNS/isaac_baseline_${STAMP}/best.pt" \
  --split validation \
  --out-dir "$OUTS/isaac_baseline_val_${STAMP}" \
  --device cuda
```

## A3. Optuna — exact same 50% as baseline

```bash
uv run python scripts/tune_gatepose_optuna.py \
  --config "$ISAAC_CFG" \
  --train-manifest splits/vanilla_seed42/train_tune_sequences.txt \
  --val-manifest splits/validation_sequences.txt \
  --run-dir "$RUNS/isaac_optuna_${STAMP}" \
  --output-dir "$OUTS/isaac_optuna_${STAMP}" \
  --trials 20 \
  --epochs-per-trial 15 \
  --device cuda
```

The important tuning artifacts are:

```text
../outputs_vanilla/isaac_optuna_0908/
├── study.db
├── trials.csv
├── best_params.json
└── best_config.yaml
```

Trial checkpoints are under:

```text
../runs_vanilla/isaac_optuna_0908/trial_0000/
../runs_vanilla/isaac_optuna_0908/trial_0001/
...
```

## A4. Final tuned training — 100% official train split, 60 epochs

```bash
uv run python scripts/train_gatepose.py \
  --config "$OUTS/isaac_optuna_${STAMP}/best_config.yaml" \
  --train-manifest splits/train_sequences.txt \
  --epochs 60 \
  --run-dir "$RUNS/isaac_final_${STAMP}" \
  --device cuda \
  --skip-final-eval
```

Only now evaluate the untouched **test** split:

```bash
uv run python scripts/evaluate_gatepose_contract.py \
  --config "$OUTS/isaac_optuna_${STAMP}/best_config.yaml" \
  --checkpoint "$RUNS/isaac_final_${STAMP}/best.pt" \
  --split test \
  --out-dir "$OUTS/isaac_final_test_${STAMP}" \
  --device cuda
```

---

# EXPERIMENT B — SYNTHETIC + ISAAC

The combined config independently opens both canonical roots and concatenates
windows from the corresponding manifest in each root.

## B1. Cheap training — 15% of each source, 8 epochs

```bash
uv run python scripts/train_gatepose.py \
  --config "$COMBINED_CFG" \
  --train-manifest splits/vanilla_seed42/train_cheap_sequences.txt \
  --epochs 8 \
  --run-dir "$RUNS/synth_isaac_cheap_${STAMP}" \
  --device cuda \
  --skip-final-eval
```

```bash
uv run python scripts/evaluate_gatepose_contract.py \
  --config "$COMBINED_CFG" \
  --checkpoint "$RUNS/synth_isaac_cheap_${STAMP}/best.pt" \
  --split validation \
  --out-dir "$OUTS/synth_isaac_cheap_val_${STAMP}" \
  --device cuda
```

## B2. Baseline training — 50% of each source, 30 epochs

```bash
uv run python scripts/train_gatepose.py \
  --config "$COMBINED_CFG" \
  --train-manifest splits/vanilla_seed42/train_baseline_sequences.txt \
  --epochs 30 \
  --run-dir "$RUNS/synth_isaac_baseline_${STAMP}" \
  --device cuda \
  --skip-final-eval
```

```bash
uv run python scripts/evaluate_gatepose_contract.py \
  --config "$COMBINED_CFG" \
  --checkpoint "$RUNS/synth_isaac_baseline_${STAMP}/best.pt" \
  --split validation \
  --out-dir "$OUTS/synth_isaac_baseline_val_${STAMP}" \
  --device cuda
```

## B3. Optuna — exact same 50% sequences as combined baseline

```bash
uv run python scripts/tune_gatepose_optuna.py \
  --config "$COMBINED_CFG" \
  --train-manifest splits/vanilla_seed42/train_tune_sequences.txt \
  --val-manifest splits/validation_sequences.txt \
  --run-dir "$RUNS/synth_isaac_optuna_${STAMP}" \
  --output-dir "$OUTS/synth_isaac_optuna_${STAMP}" \
  --trials 20 \
  --epochs-per-trial 15 \
  --device cuda
```

## B4. Final tuned training — 100% train split from BOTH sources

```bash
uv run python scripts/train_gatepose.py \
  --config "$OUTS/synth_isaac_optuna_${STAMP}/best_config.yaml" \
  --train-manifest splits/train_sequences.txt \
  --epochs 60 \
  --run-dir "$RUNS/synth_isaac_final_${STAMP}" \
  --device cuda \
  --skip-final-eval
```

Evaluate the combined untouched test sets:

```bash
uv run python scripts/evaluate_gatepose_contract.py \
  --config "$OUTS/synth_isaac_optuna_${STAMP}/best_config.yaml" \
  --checkpoint "$RUNS/synth_isaac_final_${STAMP}/best.pt" \
  --split test \
  --out-dir "$OUTS/synth_isaac_final_test_${STAMP}" \
  --device cuda
```

---

# Optional video inference using either final model

Isaac-only model:

```bash
uv run python scripts/infer_gatepose_video.py \
  --video /path/to/video.mp4 \
  --checkpoint "$RUNS/isaac_final_${STAMP}/best.pt" \
  --config "$OUTS/isaac_optuna_${STAMP}/best_config.yaml" \
  --out-dir "$OUTS/isaac_video_${STAMP}" \
  --device cuda
```

Synthetic+Isaac model:

```bash
uv run python scripts/infer_gatepose_video.py \
  --video /path/to/video.mp4 \
  --checkpoint "$RUNS/synth_isaac_final_${STAMP}/best.pt" \
  --config "$OUTS/synth_isaac_optuna_${STAMP}/best_config.yaml" \
  --out-dir "$OUTS/synth_isaac_video_${STAMP}" \
  --device cuda
```

---

# What to compare

For cheap/baseline use the validation `evaluation.json` files. For the final
comparison use only:

```text
../outputs_vanilla/isaac_final_test_0908/evaluation.json
../outputs_vanilla/synth_isaac_final_test_0908/evaluation.json
```

Each output directory also contains `per_sample_metrics.csv` for failure and
robustness analysis.

Do **not** select hyperparameters based on final test performance.

---

# Resume an interrupted normal training run

For example:

```bash
uv run python scripts/train_gatepose.py \
  --config "$ISAAC_CFG" \
  --train-manifest splits/vanilla_seed42/train_baseline_sequences.txt \
  --epochs 30 \
  --resume "$RUNS/isaac_baseline_${STAMP}/last.pt" \
  --run-dir "$RUNS/isaac_baseline_${STAMP}" \
  --device cuda \
  --skip-final-eval
```

Optuna uses persistent SQLite storage (`study.db`) and `load_if_exists=True`, so
re-running the same Optuna command resumes the named study and adds trials rather
than discarding completed trials.

---

# Reproducibility rules

- `train/validation/test` are fixed by sequence, never frame.
- Existing official split manifests are preserved.
- Cheap = existing `splits/train_cheap_sequences.txt` when valid, otherwise deterministic 15% of official train sequences.
- Baseline = deterministic 50% of official train sequences.
- Tune = the exact same sequence IDs as baseline.
- Cheap is a subset of baseline.
- Seed = 42 for subset construction, normal training, and every Optuna trial.
- Validation/test are not augmented by this compatibility layer.
- Final training uses all official **train** sequences, never validation/test.
- Combined training uses all selected synthetic + Isaac windows without source
  oversampling.

---

## Inference after each stage

Use `scripts/infer_vanilla_stage.py` for cheap, baseline, best-Optuna-trial, and final checkpoints under both data regimes. It creates a translucent segmentation overlay, keypoint/center geometry, pose/runtime HUD, overlay frames, and `overlay.mp4`.

Exact commands for `../data/refined_target/sim0721-10/video.mp4` and `../data/refined_target/sim0721-10/images` are in [`VANILLA_INFERENCE.md`](VANILLA_INFERENCE.md).
