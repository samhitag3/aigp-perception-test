# What this repo is for

**Perception model training for autonomous drone racing (ADR).** This repo
trains, fine-tunes, evaluates, and exports the **gate-perception models** a
racing drone uses to *see* the gates it flies through — at 100 km/h, fully
onboard, from a single camera.

It is the training/experimentation counterpart to the competition systems
**MonoRace** (winner of the 2025 A2RL x DCL Autonomous Drone Championship) and
**SkyDreamer**: both rely on a learned **gate segmentation** front-end, and this
repo is where that front-end is produced.

```
 raw flight footage / sim renders
            │
   ┌────────▼─────────┐   images + masks + YOLO labels + corners + intrinsics
   │ sam3-autolabeler │ ───────────────────────────────────────────────┐
   │ (SAM3 + synthetic)│                                                 │
   └──────────────────┘                                                 │
                                  scripts/import_data.py                 │
                                            │                            │
                          ┌─────────────────▼──────────────────┐        │
   THIS REPO  ─────────▶  │  data/  (YOLO + GateNet formats,     │ ◀──────┘
                          │         shared train/val/test split) │
                          └───────┬───────────────────┬─────────┘
                                  │                    │
                      train / fine-tune          evaluate (shared test set)
                                  │                    │
              ┌───────────────────┼────────────────┐   │
              ▼                   ▼                ▼   ▼
          GateNet            YOLO26-seg       YOLOE-seg   →  IoU / Dice / mAP
        (lightweight        (Ultralytics)   (open-vocab)
         U-Net, onboard)
              │
              ▼
   QuAdGate corners → PnP pose  →  (EKF + control, downstream)
```

## What's in it

| Piece | Purpose |
|-------|---------|
| **GateNet** (`gatenet/`) | Lightweight deep-supervised U-Net for binary gate segmentation. ~0.44M (f2) / ~1.75M (f4) params, ONNX/TensorRT-exportable for onboard (Jetson) deployment. The MonoRace/SkyDreamer recipe. |
| **YOLO26 / YOLOE** (`finetune/`) | Ultralytics instance-segmentation backends, fine-tuned on the same gate data. YOLOE adds open-vocabulary prompts. |
| **Perception** (`perception/`) | `QuAdGate` sub-pixel corner detector (LSD → line intersections → RANSAC) and a multi-gate **PnP** pose solver that consume the masks. |
| **Data import** (`finetune/data.py`, `scripts/import_data.py`) | Pulls datasets from the [sam3-autolabeler](https://github.com/airobotics-ucberkeley) into `data/` in **both** YOLO and GateNet formats with a shared split + camera intrinsics. |
| **Evaluation** (`evaluation/`) | One common mask IoU/Dice/PR + YOLO box/mask mAP, so every backend is scored on the identical held-out test set. |
| **Tooling** (`scripts/`, `setup.sh`) | train / finetune / evaluate / export / GPU check / one-shot server setup. |

## Why three backends?

- **GateNet** is the deployable target: tiny, fast (~3 ms on a Jetson), runs in
  the onboard control loop. It's what actually flies.
- **YOLO26 / YOLOE** are strong, well-tooled baselines for benchmarking gate
  segmentation quality and for prototyping — easy to fine-tune, but heavier.

The unified data + evaluation layer lets you compare all three apples-to-apples
and pick the right trade-off (accuracy vs. onboard latency).

## Where to go next

- **`README.md`** — quickstart, architecture, model details.
- **`TRAINING.md`** — full train / fine-tune / evaluate guide for every backend.
- **`setup.sh`** — one-shot provisioning for a GPU server (`./setup.sh --help`).
- **`data/README.md`** — the dataset contract.
- **`docs/results.md`** — GateNet vs YOLO26 benchmark on `ai_gp_gate_6000`
  (+ `docs/eval_comparison.jpg`).
- **`docs/eval_perception.jpg`** — GateNet mask → corners → orientation on real
  test data.

## Provenance

GateNet and the perception stack follow:
- Bahnam et al., *MonoRace: Winning Champion-Level Drone Racing with Robust
  Monocular AI* (A2RL 2025).
- Verraest et al., *SkyDreamer* ([arXiv:2510.14783](https://arxiv.org/abs/2510.14783)).
