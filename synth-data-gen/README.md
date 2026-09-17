# GateSynth Diverse Motion v0.3

Canonical synthetic UAV-gate sequence generator for rapid perception-model smoke testing while waiting for Isaac Sim data.

This version preserves the canonical dataset contract and adds deliberately diverse, physically time-sampled flight trajectories.

## Fixed real-camera calibration

The camera is **not randomized**. Every sequence and frame uses exactly:

```text
width  = 640
height = 360
fx = 320.0
fy = 320.0
cx = 320.0
cy = 180.0
K = [[320, 0, 320],
     [0, 320, 180],
     [0, 0, 1]]
HFoV = 90.0 deg
VFoV = 58.7155070856 deg
lens distortion = none
```

The generator fails immediately if image resolution and camera resolution disagree.

## Gate skin and geometry

The input gate skin is `SAMPLE_GATE_aigp.jpg` and the gate ring is cut from:

```python
OG_GATE_DICT = {
    "outer": [[117, 117], [906, 117], [906, 906], [117, 906]],
    "inner": [[292, 292], [731, 292], [731, 731], [292, 731]],
}
```

Physical geometry:

```text
outer width/height = 2.7 m
inner width/height = 1.5 m
depth = 0.26 m
```

## What is diverse now

Each sequence independently samples:

- a random global course direction
- 3-D gate positions
- spacing between gates
- gate altitude
- gate yaw offset relative to course direction
- gate pitch and roll
- course style: `straight`, `flowing`, `technical`, or `aggressive`
- turns between successive gates, including occasional hard turns
- lateral and vertical path offsets through the gate openings
- camera/gaze offsets over time
- motion profile: `slow`, `medium`, `fast`, or `aggressive`
- cruise speed
- acceleration limit
- explicit acceleration/deceleration speed events
- smooth speed modulation
- automatic slowing around high-curvature turns
- sequence-consistent noise tier and strengths

The trajectory is sampled at fixed FPS by integrating speed over path arc length. Therefore stored speed, velocity, acceleration, timestamps, and frame-to-frame displacement are physically consistent instead of being arbitrary spline derivatives.

## Default motion ranges

The included configs use approximately:

```text
slow       cruise 1.5-3.0 m/s
medium     cruise 3.0-5.5 m/s
fast       cruise 5.5-8.0 m/s
aggressive cruise 7.5-11.0 m/s
```

Individual frames can fall outside the cruise range because of acceleration/deceleration events and turn slowdown, subject to each profile's configured min/max speed.

## Noise/domain randomization

Noise strength is sampled once per sequence and remains consistent in severity across that sequence while the per-frame realization changes smoothly or stochastically as appropriate.

Included effects:

- brightness
- contrast
- saturation
- gamma
- color-temperature shift
- Gaussian blur
- motion blur
- Gaussian noise
- multiplicative speckle noise
- JPEG degradation
- vignette
- broad shadows
- lens flare/glare proxy

Camera calibration itself is never randomized.

## Output contract

Each dataset contains:

```text
dataset_root/
├── dataset.json
├── gate_geometry.json
├── splits/
│   ├── train_sequences.txt
│   ├── validation_sequences.txt
│   └── test_sequences.txt
└── sequences/
    ├── seq_000000/
    │   ├── sequence.json
    │   ├── frames.jsonl
    │   ├── rgb/
    │   └── instance_masks/
    └── ...
```

Per-frame records include camera pose, drone state, velocity, acceleration, target speed, motion profile, route progress, persistent gate track IDs, frame-local mask IDs, 2-D/3-D keypoints, gate pose, distance/depth, visibility, occlusion, truncation, and bounding boxes.

## Install with uv

From the repo root:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e '.[gpu]'
```

Verify the package and CUDA:

```bash
python - <<'PY'
import gatesynth
import torch
print("gatesynth:", gatesynth.__file__)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

If you intentionally want CPU-only:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

## First run: smoke test

```bash
DATE=0914
OUT="../data/synth_smoke_diverse_${DATE}"

python scripts/generate.py \
  --config configs/smoke_real_camera_diverse.yaml \
  --gate-skin ../assets/SAMPLE_GATE_aigp.jpg \
  --output "$OUT" \
  --photometric-backend auto \
  --device cuda \
  --workers 1
```

Validate it:

```bash
python scripts/validate_dataset.py "$OUT"
```

Inspect camera + trajectory diversity:

```bash
python scripts/inspect_dataset.py "$OUT"
```

A correct camera check should report exactly one K and one resolution:

```text
unique_camera_K: 1
unique_resolutions: 1
K = [[320,0,320],[0,320,180],[0,0,1]]
resolution = 640x360
```

## Large GPU-oriented run

The included large config defaults to 2,000 sequences.

```bash
DATE=0914
OUT="../data/synth_large_realcam_diverse_${DATE}"

python scripts/generate.py \
  --config configs/large_scale_gpu_real_camera_diverse.yaml \
  --gate-skin ../assets/SAMPLE_GATE_aigp.jpg \
  --output "$OUT" \
  --workers 1 \
  --photometric-backend auto \
  --device cuda \
  --frame-batch-size 32
```

The GPU mode batches the expensive photometric corruption on CUDA. Projective gate rendering and disk I/O still use CPU/OpenCV, so one GPU worker is intentional; launching several CUDA worker processes usually hurts throughput or VRAM stability.

## Generate a custom number of sequences

For 500 sequences:

```bash
python scripts/generate.py \
  --config configs/large_scale_gpu_real_camera_diverse.yaml \
  --gate-skin ../assets/SAMPLE_GATE_aigp.jpg \
  --output ../data/synth_500_diverse_0914 \
  --sequences 500 \
  --workers 1 \
  --photometric-backend auto \
  --device cuda \
  --frame-batch-size 32
```

For 5,000 sequences:

```bash
python scripts/generate.py \
  --config configs/large_scale_gpu_real_camera_diverse.yaml \
  --gate-skin ../assets/SAMPLE_GATE_aigp.jpg \
  --output ../data/synth_5000_diverse_0914 \
  --sequences 5000 \
  --workers 1 \
  --photometric-backend auto \
  --device cuda \
  --frame-batch-size 32
```

## CPU-parallel alternative

On some machines OpenCV warping/JPEG/disk I/O becomes the bottleneck rather than augmentation. Benchmark this mode too:

```bash
DATE=0914
OUT="../data/synth_large_realcam_diverse_cpu_${DATE}"

python scripts/generate.py \
  --config configs/large_scale_cpu_parallel_real_camera_diverse.yaml \
  --gate-skin ../assets/SAMPLE_GATE_aigp.jpg \
  --output "$OUT" \
  --workers 8 \
  --photometric-backend cpu \
  --device cpu
```

Do not combine many process workers with the CUDA augmentation backend unless you deliberately want several processes competing for the same GPU.

## Reproducibility

The generator is deterministic given the same global seed, sequence index, config, and code version.

Override the seed with:

```bash
python scripts/generate.py \
  --config configs/large_scale_gpu_real_camera_diverse.yaml \
  --gate-skin ../assets/SAMPLE_GATE_aigp.jpg \
  --output ../data/test_seed_123 \
  --sequences 50 \
  --seed 123
```

## Important config sections

### Course direction/placement diversity

Edit `course:`:

```yaml
course:
  gates_min: 3
  gates_max: 5
  course_spacing_m: [4.0, 8.0]
  gate_height_m: [0.9, 2.8]
  gate_yaw_deg: [-35.0, 35.0]
  gate_pitch_deg: [-15.0, 15.0]
  gate_roll_deg: [-12.0, 12.0]
```

`course_styles` controls turn severity.

### Flight-path diversity

Edit `trajectory:`:

```yaml
trajectory:
  approach_distance_m: 5.0
  exit_distance_m: 3.0
  path_lateral_offset_m: 0.55
  path_vertical_offset_m: 0.40
  camera_lookahead_m: 4.5
  target_gate_look_blend: 0.58
```

### Speed diversity

Edit `motion:`. Each profile controls:

- cruise-speed range
- acceleration limit
- speed-variation amplitude
- camera orientation jitter
- turn slowdown
- minimum/maximum speed
- start-speed fraction

## Recommended workflow

Before launching thousands of sequences:

```bash
# 1. Generate 20-50 sequences
python scripts/generate.py \
  --config configs/large_scale_gpu_real_camera_diverse.yaml \
  --gate-skin ../assets/SAMPLE_GATE_aigp.jpg \
  --output ../data/synth_benchmark \
  --sequences 25 \
  --workers 1 \
  --photometric-backend auto \
  --device cuda

# 2. Validate
python scripts/validate_dataset.py ../data/synth_benchmark

# 3. Inspect diversity
python scripts/inspect_dataset.py ../data/synth_benchmark
```

Then launch the 2,000/5,000-sequence run only after confirming the visual output and throughput.
