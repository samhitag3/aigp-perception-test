# GateSynth — sequence-consistent synthetic UAV gate dataset generator

This repository generates temporally coherent synthetic gate-flight sequences that follow the canonical UAV gate perception dataset contract.

## What it produces

- RGB JPEG frames
- one `uint16` instance-ID PNG per frame (`0=background`, `1..N=frame-local gate IDs`)
- persistent `track_id` per physical gate across a sequence
- camera intrinsics and camera/world transforms
- drone pose and velocity
- static gate world pose and camera-relative pose
- 8 physical gate keypoints (outer TL/TR/BR/BL + inner TL/TR/BR/BL)
- visible/occluded/out-of-frame keypoint state
- 3D keypoints in camera and world coordinates
- visible and amodal bounding boxes
- distance, optical depth, view angle
- visible area, occlusion fraction, truncation fraction
- current/next route target
- fixed sequence-level train/validation/test splits
- sequence-consistent photometric degradation settings
- checksums + dataset validator

## Gate skin

Pass your `SAMPLE_GATE_aigp.jpg` to `--gate-skin`.

The ring alpha mask is cut using exactly:

```python
OG_GATE_DICT = {
    "outer": [[117,117],[906,117],[906,906],[117,906]],
    "inner": [[292,292],[731,292],[731,731],[292,731]],
}
```

The image is projectively warped onto a physical gate with:

- outer: 2.7 m × 2.7 m
- inner: 1.5 m × 1.5 m
- depth: 0.26 m

## Install
Navigate to the GateSynth folder:

```bash
cd synth-data-gen
```

And set up with `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Generate a smoke-test dataset

```bash
python3 scripts/generate.py \
  --config configs/smoke.yaml \
  --gate-skin ../assets/gate_skins/SAMPLE_GATE_aigp.jpg \
  --output "../data/synth_smoke$(date +%m%d)"
```

## Validate

```bash
python3 scripts/validate_dataset.py "../data/synth_smoke$(date +%m%d)"
```

## Backgrounds

By default the generator can create procedural backgrounds. For more realism set:

```yaml
render:
  backgrounds_dir: /path/to/background/images
```

The generator chooses one background source per sequence and performs smooth temporal pan/crop so frames remain coherent.

## Why geometry is not implemented as random 2D rotations

Gate pose is generated in 3D, then projected through a pinhole camera. The course contains static gates with randomized yaw/pitch/roll, while the virtual drone follows a smooth camera trajectory through the course. This gives coherent perspective changes and trackable frame-to-frame pose.

"Zoom" is produced by varying focal length and range; "crop"/truncation is produced by camera trajectory and framing. This preserves exact geometric ground truth and camera intrinsics.

## Sequence-consistent degradation

Each sequence samples one degradation tier (`clean`, `low`, `medium`, `high`) and one fixed set of strengths for brightness, contrast, saturation, gamma, blur, motion blur, Gaussian noise, speckle noise, JPEG compression, vignette, color temperature, shadow, and flare.

The strength remains fixed for the whole trajectory; stochastic noise realization and slow lighting phase vary frame-to-frame. This avoids unrealistic abrupt domain changes while still giving temporal variety.

## Scaling for RTX 4070

The renderer is currently CPU/OpenCV based. A 4070 is useful for training but this generator emphasizes deterministic geometry and correctness. You can parallelize sequence generation safely because every sequence has an independent deterministic seed. For a large dataset, launch separate configs/ranges in parallel or add a multiprocessing wrapper around sequence generation.

Recommended first smoke test: 20–50 sequences at 640×360 or 960×540. Once model plumbing is verified, scale to thousands of sequences and higher resolution.

## Important caveats

This is a synthetic smoke-test renderer, not a replacement for Isaac Sim. It uses projective textured gate rendering and approximate painter-style depth ordering. It is intentionally designed to stress model/data interfaces while preserving exact camera/gate geometry. Isaac data should replace or augment it for final model selection.
