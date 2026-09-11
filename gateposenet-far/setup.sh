#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup.sh — provision this repo for training / fine-tuning on a GPU machine.
#
# Installs uv (if missing), creates the project venv with GPU PyTorch + the
# Ultralytics (YOLO26/YOLOE) extra, verifies CUDA, and optionally imports a
# dataset. Safe to re-run (idempotent).
#
# Usage:
#   ./setup.sh                                   # default: CUDA 12.4 + yolo + export
#   ./setup.sh --cuda cu121                      # pick a different CUDA wheel set
#   ./setup.sh --cuda cu126 --data-src /path/to/sam3-autolabeler
#   ./setup.sh --cpu                             # CPU-only (testing)
#   ./setup.sh --no-yolo                         # GateNet only, skip ultralytics
#
# Then:
#   uv run python scripts/check_gpu.py
#   uv run python scripts/train.py    --config configs/gatenet_a2rl.yaml
#   uv run --extra yolo python scripts/finetune.py --model yolo26 --variant s \
#       --data data/yolo/data.yaml --device 0
# ---------------------------------------------------------------------------
set -euo pipefail

CUDA="cu124"          # cu121 | cu124 | cu126 | cpu
WITH_YOLO=1
WITH_EXPORT=1
WITH_DEV=1
DATA_SRC=""
PYTHON_VERSION=""      # empty => use .python-version (3.10)

usage() { sed -n '2,28p' "$0"; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda)      CUDA="$2"; shift 2 ;;
    --cpu)       CUDA="cpu"; shift ;;
    --data-src)  DATA_SRC="$2"; shift 2 ;;
    --no-yolo)   WITH_YOLO=0; shift ;;
    --no-export) WITH_EXPORT=0; shift ;;
    --no-dev)    WITH_DEV=0; shift ;;
    --python)    PYTHON_VERSION="$2"; shift 2 ;;
    -h|--help)   usage ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

cd "$(dirname "$0")"
say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }

# --- 1. uv -----------------------------------------------------------------
say "1/5  uv"
if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
uv --version

# --- 2. CUDA wheel index ---------------------------------------------------
say "2/5  PyTorch wheel set: $CUDA"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
  echo "nvidia-smi not found — no NVIDIA GPU detected (CPU/MPS only)."
fi

# pyproject pins cu124 for Linux. Point the index at the requested CUDA build.
if [[ "$CUDA" == "cpu" ]]; then
  echo "CPU mode: removing the Linux CUDA torch source so CPU wheels are used."
  python3 - <<'PY'
import re, pathlib
p = pathlib.Path("pyproject.toml"); t = p.read_text()
# drop the [tool.uv.sources] torch override + the CUDA index block
t = re.sub(r"\n\[tool\.uv\.sources\].*?(?=\n\[)", "\n", t, flags=re.S)
t = re.sub(r"\n\[\[tool\.uv\.index\]\].*?(?=\n\[|\Z)", "\n", t, flags=re.S)
p.write_text(t)
print("pyproject CUDA source removed (CPU wheels).")
PY
  uv lock
elif [[ "$CUDA" != "cu124" ]]; then
  echo "rewriting CUDA index -> $CUDA"
  sed -i.bak -E "s#download\.pytorch\.org/whl/cu[0-9]+#download.pytorch.org/whl/${CUDA}#g" pyproject.toml
  sed -i.bak -E "s#(name = \")pytorch-cu[0-9]+(\")#\1pytorch-${CUDA}\2#g" pyproject.toml
  uv lock
fi

# --- 3. sync env -----------------------------------------------------------
say "3/5  uv sync"
EXTRAS=()
[[ $WITH_YOLO   -eq 1 ]] && EXTRAS+=(--extra yolo)
[[ $WITH_EXPORT -eq 1 ]] && EXTRAS+=(--extra export)
[[ $WITH_DEV    -eq 1 ]] && EXTRAS+=(--extra dev)
PYARG=(); [[ -n "$PYTHON_VERSION" ]] && PYARG=(--python "$PYTHON_VERSION")
echo "uv sync ${PYARG[*]:-} ${EXTRAS[*]:-}"
uv sync "${PYARG[@]}" "${EXTRAS[@]}"

# --- 4. verify -------------------------------------------------------------
say "4/5  verify GPU + imports"
uv run python scripts/check_gpu.py
uv run python -c "import gatenet, perception, finetune, evaluation; print('packages OK')"
if [[ $WITH_YOLO -eq 1 ]]; then
  uv run python -c "import ultralytics; print('ultralytics', ultralytics.__version__)"
fi

# --- 5. optional data import ----------------------------------------------
say "5/5  data"
if [[ -n "$DATA_SRC" ]]; then
  echo "importing datasets from: $DATA_SRC"
  uv run python scripts/import_data.py --src "$DATA_SRC"
else
  echo "skip (pass --data-src /path/to/sam3-autolabeler to import now)"
fi

cat <<'NEXT'

============================================================
 setup complete. next steps:
   # GateNet
   uv run python scripts/train.py    --config configs/gatenet_a2rl.yaml
   uv run python scripts/finetune.py --model gatenet \
       --config configs/gatenet_finetune.yaml --pretrained runs/.../best.pt
   # YOLO26 / YOLOE (GPU)
   uv run python scripts/finetune.py --model yolo26 --variant s \
       --data data/yolo/data.yaml --epochs 100 --imgsz 640 --device 0
   uv run python scripts/finetune.py --model yoloe  --variant 26s \
       --data data/yolo/data.yaml --epochs 80  --imgsz 640 --device 0
   # evaluate (shared test set)
   uv run python scripts/evaluate.py --model gatenet --config <cfg> --checkpoint <ckpt>
 see TRAINING.md for the full guide.
============================================================
NEXT
