# GateSynth Accelerated

Synthetic sequence generator for the canonical UAV gate perception dataset contract.

This accelerated version adds:

- **sequence-parallel generation** with multiple worker processes
- **optional GPU batched photometric augmentation** using PyTorch/CUDA
- large-scale configs intended for generating **many sequences**
- the same output schema as the original pipeline

## What it generates

For each sequence:

- RGB frames
- one uint16 instance-mask PNG per frame
- `sequence.json`
- `frames.jsonl`
- train/validation/test sequence manifests
- `dataset.json`
- `gate_geometry.json`

The data follows the canonical contract:

- one frame-local integer instance mask per image (`0 = background`, `1..N = gate mask IDs`)
- persistent `track_id` per gate across a sequence
- 8 canonical projected 2D keypoints per gate
- 3D keypoints, gate pose, camera pose, drone pose, distance, depth, visibility, occlusion, truncation, route metadata, etc.

## Installation

Navigate to the GateSynth folder:

```bash
cd synth-data-gen
```

CPU-only:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

With optional GPU augmentation:

```bash
uv venv
source .venv/bin/activate
uv pip install -e '.[gpu]'
```

You may also install your own CUDA-enabled PyTorch build manually.

## Gate skin

The generator expects your original `SAMPLE_GATE_aigp.jpg` and cuts out the gate ring using:

```python
OG_GATE_DICT = {
    "outer": [[117, 117], [906, 117], [906, 906], [117, 906]],
    "inner": [[292, 292], [731, 292], [731, 731], [292, 731]],
}
```

## Fast usage modes

### 1) Large-scale GPU-augmented mode

This is best when you want to exploit the RTX 4070 for batched photometric corruption. It runs one main generation process and batches the augmentations on GPU.

```bash
python scripts/generate.py \
  --config configs/large_scale_gpu.yaml \
  --gate-skin ../assets/gate_skins/SAMPLE_GATE_aigp.jpg \
  --output "../data/synth_large_$(date +%m%d)"
```

Notes:

- `generation.num_workers: 1`
- `acceleration.photometric_backend: auto`
- `acceleration.device: cuda`
- `acceleration.frame_batch_size: 32`
- `write_checksums: false` for speed

### 2) Large-scale CPU-parallel mode

This is best when sequence rendering and disk writing dominate. Multiple worker processes each render full sequences independently.

```bash
python scripts/generate.py \
  --config configs/large_scale_cpu_parallel.yaml \
  --gate-skin ../assets/gate_skins/SAMPLE_GATE_aigp.jpg \
  --output "../data/synth_large_$(date +%m%d)"
```

Notes:

- `generation.num_workers: 8` by default
- augmentation stays on CPU
- often very competitive because OpenCV warping/compositing is CPU-heavy

### 3) Quick smoke test

```bash
python scripts/generate.py \
  --config configs/smoke.yaml \
  --gate-skin ../assets/gate_skins/SAMPLE_GATE_aigp.jpg \
  --output "../data/synth_smoke_$(date +%m%d)"
```

## Command-line overrides

You can override the worker count or photometric backend directly:

```bash
python scripts/generate.py \
  --config configs/large_scale_gpu.yaml \
  --gate-skin ../assets/gate_skins/SAMPLE_GATE_aigp.jpg \
  --workers 1 \
  --photometric-backend auto \
  --device cuda
```

or:

```bash
python scripts/generate.py \
  --config configs/large_scale_cpu_parallel.yaml \
  --gate-skin ../assets/gate_skins/SAMPLE_GATE_aigp.jpg \
  --workers 12
```

## Tuning guidance

### If you want maximum total throughput

Try both:

- `large_scale_gpu.yaml`
- `large_scale_cpu_parallel.yaml`

because the bottleneck depends on your exact machine:

- if **photometric corruption** dominates, GPU mode wins
- if **OpenCV warping/compositing + JPEG writing** dominates, CPU-parallel mode may win

### Good starting values on an RTX 4070 machine

GPU mode:

- `image: 1280x720`
- `frame_batch_size: 32`
- `num_workers: 1`

CPU-parallel mode:

- `num_workers: 6` to `10`
- leave a little headroom for OS / disk I/O

## Validation

```bash
python scripts/validate_dataset.py  \
  --output "../data/synth_large_$(date +%m%d)"
```

## Speed notes

For very large runs, these settings save time:

- `write_checksums: false`
- use JPEG quality around `90-93`
- prefer SSD output
- use a real background directory only if you need it
- benchmark GPU mode vs CPU-parallel mode on ~50 sequences before launching thousands

## Output compatibility

This accelerated version preserves the same canonical dataset structure and semantics, so your future Isaac exporter can target the exact same format and all model loaders can remain unchanged.
