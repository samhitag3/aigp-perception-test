# Training & Evaluation Guide

How to train, fine-tune, and test every gate-segmentation backend in this repo:

| Backend  | What it is                              | Weights / params | Entry |
|----------|-----------------------------------------|------------------|-------|
| `gatenet`| Lightweight in-repo U-Net (MonoRace)    | ~0.44M (f2) / ~1.75M (f4) | `scripts/train.py`, `scripts/finetune.py` |
| `yolo26` | Ultralytics YOLO26 instance segmentation| `yolo26{n,s,m,l,x}-seg` | `scripts/finetune.py --model yolo26` |
| `yoloe`  | Ultralytics YOLOE open-vocab seg        | `yoloe-26{n..x}-seg` etc. | `scripts/finetune.py --model yoloe` |

All backends train on the **same imported dataset** and are evaluated on the
**same test split** (`data/test`) with a common mask IoU, so results are directly
comparable.

---

## 1. Environment

**One-shot (server / GPU box):**
```bash
./setup.sh                                   # CUDA 12.4 + yolo + export, verifies GPU
./setup.sh --cuda cu121                      # different CUDA wheel set
./setup.sh --cuda cu126 --data-src /path/to/sam3-autolabeler   # + import data
./setup.sh --cpu                             # CPU-only (testing)
```

**Manual:**
```bash
uv sync                 # GateNet + perception (torch+cu124 on Linux GPU)
uv sync --extra yolo    # adds Ultralytics for YOLO26 / YOLOE
uv sync --extra export  # adds ONNX export
uv run python scripts/check_gpu.py
```

> YOLO26 and YOLOE require `ultralytics>=8.4`. Training the YOLO backends is GPU-
> heavy; run them on the remote server. GateNet trains on CPU for small jobs.

---

## 2. Data

### Import from the autolabeler
The [sam3-autolabeler](https://github.com/airobotics-ucberkeley/sam3-autolabeler)
emits, per dataset, `images/ + labels/ (YOLO polygons) + masks/ + data.yaml +
ground_truth.json`. Import + split into `/Users/c.k./gateNET-1/data` in **both**
formats at once:

```bash
uv run python scripts/import_data.py --src /Users/c.k./sam3-autolabeler
```

Produces:
```
data/yolo/{train,val,test}/{images,labels} + data/yolo/data.yaml   # YOLO26 / YOLOE
data/synth/{images,masks}   data/real/{images,masks}               # GateNet train pool
data/test/{images,masks}                                           # shared held-out test
data/meta/{intrinsics.json, import_manifest.json}                  # camera K, manifest
```

Files are **copied** by default so `data/` is self-contained (the autolabeler
datasets are regenerated/cleaned between runs). Use `--link` to symlink instead,
`--limit-per-ds N` to cap, `--val-frac/--test-frac` to change splits.

### Offline / CI fixture (no autolabeler needed)
Generate a dataset in the exact autolabeler format, then import it:

```bash
uv run python scripts/make_mock_dataset.py --out /tmp/mock_src/gate_a --n 200
uv run python scripts/import_data.py --src /tmp/mock_src
```

---

## 3. GateNet — train from scratch

```bash
# on the imported autolabeler data
uv run python scripts/train.py --config configs/gatenet_synth.yaml
# the A2RL/MonoRace recipe (384, f=4) or SkyDreamer variant (196, f=2):
uv run python scripts/train.py --config configs/gatenet_a2rl.yaml
uv run python scripts/train.py --config configs/gatenet_mavlab.yaml
```

Checkpoints → `runs/<name>/{best,last}.pt`. TensorBoard logs under `runs/<name>/tb`.

## 4. GateNet — fine-tune to a new gate / venue

```bash
uv run python scripts/finetune.py --model gatenet \
    --config configs/gatenet_finetune.yaml --pretrained runs/gatenet_synth/best.pt
# regimes:
#   --freeze-encoder     train decoder + heads only (tiny data)
#   --reset-heads        re-init output heads (very different appearance)
#   --freeze-bn-stats    freeze encoder BatchNorm running stats
```

## 5. YOLO26 — fine-tune

```bash
uv run --extra yolo python scripts/finetune.py --model yolo26 --variant s \
    --data data/yolo/data.yaml --epochs 100 --imgsz 640 --batch 16 --device 0
```
Outputs to `yolo26/<run>/weights/best.pt`. Variants: `n,s,m,l,x`.

## 6. YOLOE — fine-tune (open-vocabulary)

```bash
uv run --extra yolo python scripts/finetune.py --model yoloe --variant 26s \
    --data data/yolo/data.yaml --epochs 80 --imgsz 640 --batch 16 --device 0 \
    --classes gate
```
Outputs to `yolo26e/<run>/weights/best.pt`. Uses `YOLOEPESegTrainer`. Variants:
`26n,26s,26m,26l,26x, 11s,11m,11l, v8s,v8m,v8l`.

---

## 7. Evaluate (unified, shared test set)

Every backend reports IoU / Dice / precision / recall / F1 on `data/test`;
YOLO also reports box/mask mAP.

```bash
# GateNet
uv run python scripts/evaluate.py --model gatenet \
    --config configs/gatenet_synth.yaml --checkpoint runs/gatenet_synth/best.pt \
    --json results_gatenet.json

# YOLO26
uv run --extra yolo python scripts/evaluate.py --model yolo26 \
    --weights yolo26/<run>/weights/best.pt --data data/yolo/data.yaml \
    --json results_yolo26.json

# YOLOE
uv run --extra yolo python scripts/evaluate.py --model yoloe \
    --weights yolo26e/<run>/weights/best.pt --data data/yolo/data.yaml \
    --json results_yoloe.json
```

---

## 8. Export GateNet for onboard deployment

```bash
uv run python scripts/export_onnx.py --config configs/gatenet_a2rl.yaml \
    --checkpoint runs/gatenet_a2rl/best.pt --output gatenet.onnx
# Jetson:  trtexec --onnx=gatenet.onnx --saveEngine=gatenet.plan --fp16
```
YOLO exports via Ultralytics: `yolo export model=yolo26/<run>/weights/best.pt format=onnx`.

---

## 9. Quick end-to-end CPU smoke (no GPU, no autolabeler)

```bash
uv run python scripts/make_mock_dataset.py --out /tmp/mock_src/gate --n 120
uv run python scripts/import_data.py --src /tmp/mock_src
uv run python scripts/train.py    --config configs/gatenet_synth.yaml --epochs 5 --device cpu
uv run python scripts/evaluate.py --model gatenet --config configs/gatenet_synth.yaml \
    --checkpoint runs/gatenet_synth/best.pt --device cpu
uv run python -m pytest tests/ -q
```

---

## Tests

```bash
uv run python -m pytest tests/ -q     # model, quadgate/pnp, finetune, data import
```
