#!/usr/bin/env bash
set -euo pipefail

# Run visual inference for every trained vanilla stage in both data regimes.
# Usage:
#   bash scripts/infer_all_vanilla_stages.sh
#   bash scripts/infer_all_vanilla_stages.sh /path/to/video.mp4 0908

INPUT="${1:-../data/refined_target/sim0721-10/video.mp4}"
STAMP="${2:-$(date +%m%d)}"
DEVICE="${DEVICE:-cuda}"

for REGIME in isaac synth_isaac; do
  for STAGE in cheap baseline optuna final; do
    echo "============================================================"
    echo "Inference: ${REGIME} / ${STAGE} / ${STAMP}"
    echo "============================================================"
    uv run python scripts/infer_vanilla_stage.py \
      --regime "$REGIME" \
      --stage "$STAGE" \
      --input "$INPUT" \
      --stamp "$STAMP" \
      --device "$DEVICE"
  done
done
