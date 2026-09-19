# Vanilla GatePoseNet inference — every training stage

This repository uses one visual inference engine for all checkpoints:

```bash
scripts/infer_gatepose.py
```

and a stage resolver:

```bash
scripts/infer_vanilla_stage.py
```

The resolver automatically finds the correct checkpoint/config for:

- `cheap`
- `baseline`
- best `optuna` trial
- `final`

for either:

- `isaac`
- `synth_isaac`

## Recommended input

Vanilla GatePoseNet is temporal (ConvGRU), so the video is preferred:

```text
../data/refined_target/sim0721-10/video.mp4
```

The image folder is also supported:

```text
../data/refined_target/sim0721-10/images
```

For an image folder, files are naturally sorted and processed as one continuous sequence. Use `--image-fps` to specify the sequence FPS (default 30).

---

## Setup

From the repository root:

```bash
STAMP=$(date +%m%d)
INPUT=../data/refined_target/sim0721-10/video.mp4
```

On September 8, `STAMP=0908`.

---

# Isaac-only inference

## Cheap

```bash
uv run python scripts/infer_vanilla_stage.py \
  --regime isaac \
  --stage cheap \
  --input "$INPUT" \
  --stamp "$STAMP" \
  --device cuda
```

Output:

```text
../outputs_vanilla/isaac_cheap_infer_0908/
```

## Baseline

```bash
uv run python scripts/infer_vanilla_stage.py \
  --regime isaac \
  --stage baseline \
  --input "$INPUT" \
  --stamp "$STAMP" \
  --device cuda
```

Output:

```text
../outputs_vanilla/isaac_baseline_infer_0908/
```

## Best Optuna trial

```bash
uv run python scripts/infer_vanilla_stage.py \
  --regime isaac \
  --stage optuna \
  --input "$INPUT" \
  --stamp "$STAMP" \
  --device cuda
```

The resolver reads:

```text
../outputs_vanilla/isaac_optuna_0908/best_params.json
```

and automatically selects:

```text
../runs_vanilla/isaac_optuna_0908/trial_XXXX/best.pt
```

for the best trial.

Output:

```text
../outputs_vanilla/isaac_optuna_infer_0908/
```

## Final

```bash
uv run python scripts/infer_vanilla_stage.py \
  --regime isaac \
  --stage final \
  --input "$INPUT" \
  --stamp "$STAMP" \
  --device cuda
```

Output:

```text
../outputs_vanilla/isaac_final_infer_0908/
```

---

# Synthetic + Isaac inference

## Cheap

```bash
uv run python scripts/infer_vanilla_stage.py \
  --regime synth_isaac \
  --stage cheap \
  --input "$INPUT" \
  --stamp "$STAMP" \
  --device cuda
```

Output:

```text
../outputs_vanilla/synth_isaac_cheap_infer_0908/
```

## Baseline

```bash
uv run python scripts/infer_vanilla_stage.py \
  --regime synth_isaac \
  --stage baseline \
  --input "$INPUT" \
  --stamp "$STAMP" \
  --device cuda
```

Output:

```text
../outputs_vanilla/synth_isaac_baseline_infer_0908/
```

## Best Optuna trial

```bash
uv run python scripts/infer_vanilla_stage.py \
  --regime synth_isaac \
  --stage optuna \
  --input "$INPUT" \
  --stamp "$STAMP" \
  --device cuda
```

The resolver automatically reads the best trial from:

```text
../outputs_vanilla/synth_isaac_optuna_0908/best_params.json
```

Output:

```text
../outputs_vanilla/synth_isaac_optuna_infer_0908/
```

## Final

```bash
uv run python scripts/infer_vanilla_stage.py \
  --regime synth_isaac \
  --stage final \
  --input "$INPUT" \
  --stamp "$STAMP" \
  --device cuda
```

Output:

```text
../outputs_vanilla/synth_isaac_final_infer_0908/
```

---

# Image-folder version

Replace the video with:

```bash
INPUT=../data/refined_target/sim0721-10/images
```

and, if the source sequence was 60 FPS, add:

```bash
--image-fps 60
```

Example:

```bash
uv run python scripts/infer_vanilla_stage.py \
  --regime isaac \
  --stage final \
  --input ../data/refined_target/sim0721-10/images \
  --image-fps 60 \
  --stamp "$STAMP" \
  --device cuda
```

---

# Visual output

Every inference directory contains:

```text
<output>/
├── inference.json
├── frames.jsonl
├── overlay.mp4
├── overlays/
│   ├── frame_000000.jpg
│   ├── frame_000001.jpg
│   └── ...
└── union_masks/
    ├── frame_000000.png
    ├── frame_000001.png
    └── ...
```

The output frame includes:

- translucent predicted gate segmentation overlay
- four predicted outer corners
- predicted gate center
- stage/regime label
- target detection confidence
- predicted visible fraction
- predicted mask area percentage
- predicted camera-relative `x/y/z`
- predicted range/depth
- roll/pitch/yaw visualization
- inference latency
- total pipeline latency
- instantaneous throughput/FPS

Green overlay means the target confidence exceeds `--presence-th`. Orange means the model produced a mask/pose but target confidence is below the accepted threshold.

These on-frame values are **prediction/runtime statistics, not accuracy metrics**, because the target video/image directory is being used as unlabeled inference input. Ground-truth accuracy metrics come from `scripts/evaluate_gatepose_contract.py`.

---

# Useful visualization controls

Increase/decrease transparency:

```bash
--overlay-alpha 0.38
```

Change target presence threshold:

```bash
--presence-th 0.5
```

Change segmentation threshold:

```bash
--mask-th 0.5
```

Smoke-test only the first 100 frames:

```bash
--max-frames 100
```

Display a live OpenCV window (only on a machine with a GUI):

```bash
--show
```

On a headless SSH/server session, omit `--show`; use `overlay.mp4` instead.

---

# Generic/manual inference

You can bypass stage resolution and provide a checkpoint directly:

```bash
uv run python scripts/infer_gatepose.py \
  --input ../data/refined_target/sim0721-10/video.mp4 \
  --checkpoint ../runs_vanilla/isaac_final_0908/best.pt \
  --config ../outputs_vanilla/isaac_optuna_0908/best_config.yaml \
  --out-dir ../outputs_vanilla/manual_infer_0908 \
  --label "isaac | final | 0908" \
  --device cuda
```

---

# Run all 8 trained checkpoints at once

After cheap, baseline, Optuna, and final training have all completed for both regimes:

```bash
bash scripts/infer_all_vanilla_stages.sh \
  ../data/refined_target/sim0721-10/video.mp4 \
  0908
```

This runs:

```text
isaac / cheap
isaac / baseline
isaac / optuna
isaac / final
synth_isaac / cheap
synth_isaac / baseline
synth_isaac / optuna
synth_isaac / final
```

and creates one dated output directory for each checkpoint under `../outputs_vanilla/`.
