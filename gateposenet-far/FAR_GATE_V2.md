# GatePoseNet-MG Far-Gate v2

This variant targets poor recall on small/distant background gates while preserving the overall GatePoseNet-MG query/pose design.

## Changes

1. `pose_sup_max_m` is raised from 25 m to 50 m, so distant gates are not silently removed from training.
2. A second ConvGRU state runs at 1/8 image resolution. The original temporal state remains at 1/16.
3. Transformer queries cross-attend to both temporal resolutions (1/16 + 1/8), improving small-object presence evidence.
4. Instance masks are decoded/supervised at 1/2 resolution instead of 1/4.
5. Thin masks use area downsampling + occupancy preservation so distant gate bars do not vanish in GT.
6. Presence/mask/corner/pose supervision is weighted toward visible far/small gates.
7. Presence uses focal modulation to emphasize hard missed gates.
8. Fixed a presence-target scatter bug where invalid padded GT entries could overwrite a real query-0 positive.
9. Validation reports `gate_recall`, `far_gate_recall`, and `far_mask_iou`.
10. Best-checkpoint selection uses a far-recall-aware composite instead of only target-gate pose error.

Legacy MG configs still instantiate the legacy architecture/checkpoint layout because all new model flags default off.

## Recommended cheap comparison

```bash
STAMP=$(date +%m%d)
RUNS=../runs_vanilla
OUTS=../outputs_vanilla
MG_COMBINED_FAR=configs/mg/combined_far.yaml

uv run python scripts/train_gatepose_mg_contract.py \
  --config "$MG_COMBINED_FAR" \
  --train-manifest splits/vanilla_seed42/train_cheap_sequences.txt \
  --epochs 8 \
  --run-dir "$RUNS/mg_synth_isaac_far_cheap_${STAMP}" \
  --device cuda
```

The progress line now includes `recall`, `far_recall`, and `far_iou`.

## Baseline

```bash
uv run python scripts/train_gatepose_mg_contract.py \
  --config "$MG_COMBINED_FAR" \
  --train-manifest splits/vanilla_seed42/train_baseline_sequences.txt \
  --epochs 30 \
  --run-dir "$RUNS/mg_synth_isaac_far_baseline_${STAMP}" \
  --device cuda
```

## Tune

```bash
uv run python scripts/tune_gatepose_mg_optuna.py \
  --config "$MG_COMBINED_FAR" \
  --train-manifest splits/vanilla_seed42/train_tune_sequences.txt \
  --val-manifest splits/validation_sequences.txt \
  --run-dir "$RUNS/mg_synth_isaac_far_optuna_${STAMP}" \
  --output-dir "$OUTS/mg_synth_isaac_far_optuna_${STAMP}" \
  --trials 20 \
  --epochs-per-trial 15 \
  --device cuda
```

## Final

```bash
uv run python scripts/train_gatepose_mg_contract.py \
  --config "$OUTS/mg_synth_isaac_far_optuna_${STAMP}/best_config.yaml" \
  --train-manifest splits/train_sequences.txt \
  --epochs 60 \
  --run-dir "$RUNS/mg_synth_isaac_far_final_${STAMP}" \
  --device cuda
```

For Isaac-only experiments, replace `combined_far.yaml` with `isaac_far.yaml` and use `mg_isaac_far_*` run names.
