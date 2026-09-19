# MaskPoseNet-MG

Mask-only GatePoseNet-MG variant. The network still receives 3 channels, but they are a colored rendering of integer per-gate instance masks, never RGB imagery. This preserves the existing encoder/ConvGRU/query-decoder architecture. Training randomly reassigns gate colors per temporal window so color does not become a semantic gate ID.

## Setup
```bash
cd maskposenet
uv sync
source .venv/bin/activate

DATE=0914
SYNTH=../data/synth_large_0906
ISAAC=../data/refined_isaac_0908
RUNS=../runs_mask
OUTS=../outputs_mask
CFG=configs/maskposenet/combined.yaml
mkdir -p "$RUNS" "$OUTS"
```

## Prepare deterministic sequence subsets
```bash
uv run python scripts/prepare_vanilla_splits.py --dataset-root "$SYNTH" --seed 42 --cheap-fraction 0.15 --baseline-fraction 0.50
uv run python scripts/prepare_vanilla_splits.py --dataset-root "$ISAAC" --seed 42 --cheap-fraction 0.15 --baseline-fraction 0.50
```
If official train/validation/test manifests are missing, add `--create-base-if-missing` only if you intentionally want this repo to create the 80/10/10 split.

## Validate mask datasets
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

## Optuna — exact same 50%, 20 trials × 15 epochs
The search includes temporal training window `[4,6,8,10,12,16]`. Deployment is stateful `model.step()`, so window length does not increase per-frame inference latency; it does increase training VRAM/time. A tiny tie-break reward prefers larger windows when accuracy is essentially tied. Each trial also benchmarks `step()` and can be pruned by the configured 5090 proxy latency budget.

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

## Final — 100%, 60 epochs using Optuna best config
```bash
uv run python scripts/train_maskposenet.py \
  --config "$OUTS/optuna_${DATE}/best_config.yaml" \
  --train-manifest splits/train_sequences.txt \
  --epochs 60 \
  --run-dir "$RUNS/final_${DATE}" \
  --device cuda

uv run python scripts/evaluate_maskposenet.py \
  --config "$OUTS/optuna_${DATE}/best_config.yaml" \
  --checkpoint "$RUNS/final_${DATE}/best.pt" \
  --split test \
  --out-dir "$OUTS/final_test_${DATE}" \
  --device cuda

uv run python scripts/benchmark_maskposenet.py \
  --config "$OUTS/optuna_${DATE}/best_config.yaml" \
  --checkpoint "$RUNS/final_${DATE}/best.pt" \
  --device cuda \
  --out "$OUTS/final_latency_${DATE}.json"
```

## Inference
Preferred input is a folder of integer instance PNGs: background=0, each gate=1..N. RGB is optional and visualization-only.

Mask-only visualization:
```bash
uv run python scripts/infer_maskposenet.py \
  --input "$ISAAC/sequences/env_000_episode_0000001/instance_masks" \
  --checkpoint "$RUNS/final_${DATE}/best.pt" \
  --config "$OUTS/optuna_${DATE}/best_config.yaml" \
  --out-dir "$OUTS/final_infer_${DATE}" \
  --fps 60 \
  --device cuda
```

With RGB only as the display background:
```bash
uv run python scripts/infer_maskposenet.py \
  --input "$ISAAC/sequences/env_000_episode_0000001/instance_masks" \
  --rgb-input "$ISAAC/sequences/env_000_episode_0000001/rgb" \
  --checkpoint "$RUNS/final_${DATE}/best.pt" \
  --config "$OUTS/optuna_${DATE}/best_config.yaml" \
  --out-dir "$OUTS/final_infer_rgb_${DATE}" \
  --fps 60 \
  --device cuda
```

Outputs include `overlay.mp4`, per-frame overlays, predicted uint16 instance masks, `frames.jsonl`, and `inference.json` with GPU model latency, end-to-end latency, effective Hz, sampled GPU utilization when available, and peak CUDA memory.
