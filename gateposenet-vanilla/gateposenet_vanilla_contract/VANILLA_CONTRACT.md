# Vanilla GatePoseNetSingle + Canonical Dataset Contract

This repository preserves the **original/legacy GatePoseNetSingle architecture** and adapts the standardized UAV gate-perception dataset around it.

## Vanilla guarantee

`gateposenet/model_single.py` was not changed.

SHA-256 of the uploaded original and this repository copy:

```text
8bd32c8548e00b699fcdbf7d913b43f2a54f8b449756ad152adeac32086c633b
```

The original model therefore still has the same:

- GateNet-style CNN encoder
- ego-motion MLP
- ConvGRU temporal state
- 1/4-resolution **one-channel union segmentation** auxiliary decoder
- four outer-corner head
- center head
- camera-relative position head
- log-depth head
- 6D rotation head
- target visibility / visible-fraction heads
- auxiliary focal-scale head
- loss functions and training-selection metric

No multi-gate query decoder, inner-corner head, instance-mask head, or track-ID head was added.

## Camera model

The project camera is fixed to:

```text
native image: 640 x 360
fx = 320 px
fy = 320 px
cx = 320 px
cy = 180 px
```

These intrinsics imply:

```text
HFoV = 2 atan(640 / (2*320)) = 90.000 deg
VFoV = 2 atan(360 / (2*320)) = 58.716 deg
```

So the previously supplied `VFoV = 90 deg` label is treated as a mislabeled **HFoV**.

The legacy model input remains exactly the original configured `320 x 192` tensor size. The loader resizes the native 640x360 RGB frame to that size, matching the original repository behavior.

## How the canonical dataset maps into the vanilla model

The source dataset remains rich and multi-gate. The compatibility adapter selects only what the original model can consume:

| Canonical source | Vanilla GatePoseNet input/target |
|---|---|
| RGB frame | RGB tensor |
| integer instance-ID PNG (`0,1,2,...`) | `instance_mask > 0` union auxiliary mask |
| all gate records | current target gate only for pose/keypoint regression |
| 8 projected keypoints | outer TL/TR/BR/BL only |
| `T_camera_gate` | `position_cam` + `R_cam_gate` |
| camera K | original `k_scale` auxiliary target |
| canonical pose/timestamps | recurrent ego input `[v_cam, omega_cam, dt]` |
| sequence split manifests | rolling windows constrained to one sequence/split |

The canonical dataset is **not modified or down-converted on disk**.

### Target gate selection

For each frame the adapter uses, in order:

1. `frame.route.current_target_track_id`
2. `gate.is_current_target == true`
3. `frame.route.current_target_route_index`
4. deterministic first route gate fallback

If the current target is temporarily fully occluded and is not in the frame's `gates[]`, the adapter can reconstruct its camera-relative GT from `T_world_camera`, the sequence `T_world_gate`, and `gate_geometry.json`.

### Segmentation semantics

Canonical mask:

```text
0 = background
1 = gate instance A
2 = gate instance B
3 = gate instance C
...
```

Vanilla model target:

```python
union_mask = instance_mask > 0
```

This is required because the original architecture has one segmentation channel. It does **not** become an instance-segmentation network.

### Keypoint semantics

The canonical dataset contains 8 keypoints:

```text
outer_tl outer_tr outer_br outer_bl
inner_tl inner_tr inner_br inner_bl
```

The original model consumes only:

```text
outer_tl outer_tr outer_br outer_bl
```

The compatibility adapter intentionally uses the original legacy normalized-coordinate convention:

```text
u_legacy = x_px / image_width
v_legacy = y_px / image_height
```

rather than changing the old model/loss to the canonical convenience normalization using `W-1` / `H-1`.

## New compatibility files

```text
gateposenet/canonical_dataset.py
    canonical-data -> unchanged GatePoseNetSingle tensor adapter

gateposenet/contract_reporting.py
    writes standardized evaluation.json + per_sample_metrics.csv

configs/gateposenet_vanilla_contract.yaml
    canonical-data training config with AIGP camera settings

scripts/validate_dataset.py
    validates schema, sequence split leakage, images, instance masks,
    track/mask IDs, transforms, keypoints, K, and optional reprojection

scripts/evaluate_gatepose_contract.py
    standardized evaluation on train/validation/test split

scripts/infer_gatepose_video.py
    standardized video inference output
```

## Validate a dataset

```bash
uv run python scripts/validate_dataset.py \
  --dataset-root ../data/perception_dataset_v1
```

Expected camera summary:

```text
AIGP intrinsics imply HFoV=90.000 deg, VFoV=58.716 deg
nominal K = [[320,0,320],[0,320,180],[0,0,1]]
```

## Train

```bash
uv run python scripts/train_gatepose.py \
  --config configs/gateposenet_vanilla_contract.yaml \
  --dataset-root ../data/perception_dataset_v1 \
  --run-dir runs/gateposenet_vanilla_contract \
  --device cuda
```

Optional manifest overrides:

```bash
--train-manifest splits/train_sequences.txt
--val-manifest splits/validation_sequences.txt
--test-manifest splits/test_sequences.txt
```

At the end of canonical-data training, the script automatically evaluates `best.pt` on the **validation** manifest and writes:

```text
runs/gateposenet_vanilla_contract/
├── best.pt
├── last.pt
├── evaluation.json
└── per_sample_metrics.csv
```

Use `--skip-final-eval` only when intentionally skipping the standard report. The test split is deliberately untouched during normal model selection. For an intentional final test evaluation, either run `scripts/evaluate_gatepose_contract.py --split test` or train with `--final-eval-split test`.

## Evaluate an existing checkpoint

```bash
uv run python scripts/evaluate_gatepose_contract.py \
  --config configs/gateposenet_vanilla_contract.yaml \
  --dataset-root ../data/perception_dataset_v1 \
  --checkpoint runs/gateposenet_vanilla_contract/best.pt \
  --split test \
  --device cuda
```

The report honestly marks unsupported vanilla metrics as unavailable. For example:

- union segmentation metrics: available
- per-instance AP/count accuracy: unavailable
- outer-corner error/PCK: available
- inner-corner metrics: unavailable
- translation/rotation/depth: available
- persistent track-ID metrics: unavailable

## Video inference

```bash
uv run python scripts/infer_gatepose_video.py \
  --video flight.mp4 \
  --checkpoint runs/gateposenet_vanilla_contract/best.pt \
  --config configs/gateposenet_vanilla_contract.yaml \
  --out-dir outputs/flight \
  --device cuda
```

The original model expects ego motion. If only video is supplied, the script uses zero linear/angular ego motion plus the exact video-frame `dt`, matching the model's modality-dropout fallback behavior.

If camera-frame ego estimates are available, provide JSONL:

```bash
--ego-jsonl flight_ego.jsonl
```

with records such as:

```json
{"frame_index":42,"ego_motion_cam":[0.1,0.0,2.3,0.01,-0.02,0.03]}
```

Video output:

```text
outputs/flight/
├── inference.json
├── frames.jsonl
└── union_masks/
    ├── frame_000000.png
    └── ...
```

Because the architecture is vanilla, `inference.json` explicitly declares:

```text
union_segmentation = true
instance_segmentation = false
outer_keypoints = true
inner_keypoints = false
pose = true
tracking = false
```

No fake per-gate instances are emitted from a model that does not predict them.

## Original export remains usable

The existing ONNX export still operates on the unchanged model:

```bash
uv run python scripts/export_gatepose.py \
  --config configs/gateposenet_vanilla_contract.yaml \
  --checkpoint runs/gateposenet_vanilla_contract/best.pt \
  --out runs/gateposenet_vanilla_contract/gateposenet_step.onnx \
  --device cuda
```

## Important limitation for model comparison

This vanilla model is a baseline for the original single-target approach. It should **not** be credited with capabilities that only the richer data contract makes possible.

In particular, it cannot natively produce:

- one mask per visible gate
- multiple gate poses per frame
- 8-keypoint predictions per gate
- persistent gate identities

Those require an architectural change and should be evaluated as separate candidate models.

## Multi-source training compatibility

The compatibility layer also supports canonical multi-source configs via
`data.sources`. This does **not** change `GatePoseNetSingle`; it only concatenates
canonical sequence windows before batching.

```yaml
data:
  sources:
    - name: synthetic
      root: ../data/synth_large_0906
    - name: isaac
      root: ../data/refined_isaac_0908
  train_manifest: splits/train_sequences.txt
  val_manifest: splits/validation_sequences.txt
  test_manifest: splits/test_sequences.txt
```

Each relative manifest is resolved independently inside each source root. Thus a
combined training run uses synthetic-train + Isaac-train, combined validation
uses synthetic-validation + Isaac-validation, and combined final test uses the
two untouched test sets. No frames or sequences are copied between roots.

See `VANILLA_TRAINING.md` for the fixed deterministic cheap/baseline/tuning/final
protocol.
