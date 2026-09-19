# OVERVIEW — what we train here, why, and how the data is made

This repo trains the **perception stack for AI Grand Prix drone racing**:
estimating where the next gate is, relative to the drone's camera, from a
single forward FPV stream — fast enough for a 30 Hz control loop, robust
enough for high-speed racing under deep uncertainty (partial views, blind
fly-through moments, occluding gates, arbitrary attitude including inverted
flight, imperfect calibration).

Training is **(nearly) entirely synthetic**. Every dataset is rendered by the
sibling repo `sam3-autolabeler-private` (`python -m synthetic gen …`), whose
labels are *analytic* — computed from known geometry, not predicted — so the
supervision is exact by construction. The generation philosophy is documented
in `sam3-autolabeler-private/synthetic/when-generating-data.md`; per-model
generation commands are given below.

Shared ground truth of the physical target (gate dimensions, camera spec,
frame conventions): **`configs/gate_aigp.yaml`** — the single place these
numbers live (source: VADR-TS-002 v00.02 §3.7–3.8).

---

## The three model families

### 1. GateNet — binary gate segmentation (`gatenet/`)

**What.** A small U-Net (double 3×3 conv blocks, additive skip connections,
deep supervision over 5 scales, widths scaled by a factor *f*) that maps an
RGB frame to a gate-frame mask. ~0.1–1.5 M params depending on *f*.

**Why.** This is the MonoRace/SkyDreamer front end: a mask is the cheapest
robust intermediate representation of "gate-ness". MonoRace (winner, A2RL×DCL
2025) feeds the mask to classical corner extraction + PnP; SkyDreamer feeds a
64×64 mask directly to a world-model policy. We reimplement it (a) as the
proven baseline, (b) as the perception front end for mask-based policies,
and (c) because its encoder is the backbone of GatePoseNet below.

**Train / eval.**
```bash
uv run python scripts/train.py --config configs/gatenet_synth.yaml
uv run python scripts/eval_perception.py ...   # mask -> corners -> PnP path
```

**Data generation (sam3-autolabeler-private).** Segmentation only needs
2-D-exact labels, so the full geometric realism menu is allowed (lens
distortion, perspective jitter, affine — they warp image+mask+corners
consistently):
```bash
python -m synthetic gen --preset aigp --num 8000 --bg-mode random \
    --hsv --brightness-gradient --affine-aug \
    --out data/synthetic_output/aigp_seg_train
# import into this repo's images/ + masks/ contract:
uv run python scripts/import_data.py
```

### 2. YOLO26 / YOLOE backends (`finetune/`, `yolo26/`)

**What.** Ultralytics segmentation models fine-tuned on the same data via the
backend registry (`finetune/backends/`). Heavier and slower than GateNet but
stronger detectors in cluttered scenes; used for offline auto-labeling and as
a cross-check on GateNet.

**Data generation.** Same datasets as GateNet — the sam3 exporter writes YOLO
polygon labels (`labels/*.txt` + `data.yaml`) alongside the masks in every
run, so no extra generation step is needed:
```bash
uv run python scripts/finetune.py --backend yolo26 ...
```

### 3. GatePoseNet — temporal multimodal gate-pose regression (`gateposenet/`)

**What.** The new model this repo exists to produce: a **1.46 M-param**
rolling-window network that outputs, per 30 Hz frame, in the **camera optical
frame**:

| head | output | downstream use |
|---|---|---|
| corners (4×2, image-normalized) + per-corner in-frame flags | 2-D keypoints | corners→PnP when calibrated (MonoRace path); partial corner sets usable |
| center (2) | fly-through point in the image | steering cue |
| position (3, meters) + log-depth | direct metric gate position | uncalibrated / degraded path — the gate's known 2.7 m size makes scale observable without K |
| 6-D rotation → R_cam_gate | gate orientation; fly-through axis = R[:,2] | approach alignment |
| visibility logit + visible fraction | trust gating | when to believe the 2-D heads |
| aux: segmentation (¼ res), fx/fy self-calibration | training-time regularizers | SkyDreamer-style privileged decoding |

**Architecture.** GateNet encoder (width factor 2) per frame → ego-motion
embedding ([v, ω] in camera frame; the FC's EKF at deployment, exact values in
sim; 25 % modality dropout so it degrades gracefully when absent) **added at
the bottleneck** → one **ConvGRU** cell carries the temporal state → pooled
state → multi-task MLP head; light decoder for the aux mask.
`model.step(image, ego, h)` is the deployment API; `forward` scans training
windows (T=8).

**Why temporal.** During a fast fly-through the gate leaves the frame — first
slivers, then nothing — exactly when the pose matters most. MonoRace coasts
on an IMU EKF through this; SkyDreamer carries it in a GRU latent. We take
the learned route and additionally **supervise the 3-D pose through the blind
frames** (down-weighted): with ego-motion input this is well-posed, and it is
what teaches the recurrent state to integrate motion while blind. Validated:
blind-frame median position error 0.91 m vs 0.56 m visible.

**Why symmetry-aware supervision.** The gate is a plain square ring — a
**D4-symmetric** object (4 rotations about the fly-through axis × front/back
flip = 8 visually identical orientations). Over full-attitude training data
the raw world-derived orientation label is ambiguous: identical images,
conflicting labels — textbook label noise. `GatePoseLoss(symmetry_aware=true)`
scores rotation as the min over the 8 symmetry-equivalent GT rotations and
corners over the 8 induced label permutations (position, center, fly-through
axis up to sign, and visibility are symmetry-invariant). Standard practice for
symmetric-object pose estimation.

**Why camera-frame outputs.** Racing camera mounts are tilted up anywhere
from ~15–60°. All outputs are in the camera optical frame, which is
tilt-invariant; the camera→body conversion is one known rotation applied
downstream (`synthetic.pose_gt.optical_to_body(tilt_deg)` in the sam3 repo).
The tilt *is* modeled in the data (viewpoint distribution + recorded per
sequence as `camera_tilt_deg`).

**Why only ±4 % intrinsics jitter (not "camera-agnostic").** For a known-size
gate, focal length and depth are exactly confounded (apparent size ∝
f·size/z). Wide focal randomization does not buy generality — it injects an
irreducible depth ambiguity the network can only average over. We therefore
randomize K per sequence only within the honest uncertainty about the
deployment camera (±4 % focal, ±1.5 % principal point), and normalize 2-D
targets by image size, never by K. Consequence measured at eval: PnP on
ground-truth corners with the nominal K bottoms out at ~0.10 m — the floor set
by the jitter itself — and the direct metric head matches PnP on predicted
corners without using any calibration.

**Train / eval / try it.**
```bash
uv run python scripts/train_gatepose.py --config configs/gateposenet_traj.yaml
uv run python scripts/eval_gatepose.py  --config configs/gateposenet_traj.yaml \
    --checkpoint runs/gateposenet_traj/best.pt
# drop images into data/test_images/ then:
uv run python scripts/run_test_images.py
```

**Data generation (sam3-autolabeler-private).** GatePoseNet trains on
*trajectory sequences* with `--pose-safe` (see the why in
`when-generating-data.md` — 2-D image warps break the image↔3-D-pose pinhole
relation, so geometric variety comes from real camera-pose randomization
instead):
```bash
python -m synthetic gen --preset aigp --trajectories 160 --pose-safe \
    --bg-mode random --formats yolo_seg,masks_png --seed 0 \
    --out data/synthetic_output/aigp_traj_train
python -m synthetic gen --preset aigp --trajectories 28  --pose-safe \
    --bg-mode random --formats yolo_seg,masks_png --seed 1000 \
    --out data/synthetic_output/aigp_traj_val
# audit what you generated (viewpoints/attitude/layouts/occlusion figures):
python -m synthetic report --dataset data/synthetic_output/aigp_traj_train
```
This produces 30 Hz fly-through / hover / passby sequences with: full-sphere
viewpoints and ±180° roll (inverted flight), multi-gate scenes (clusters and
A2RL-style stacked towers, 3 m minimum center spacing, occlusion-corrected
per-gate masks), per-sequence photometric profiles + lighting DR, exact
ego-motion, per-sequence camera tilt (15–60°) and intrinsics jitter, and
blind frames carried with exact pose GT.

**Reference results** (held-out val, 2,628 frames, symmetry-aware metrics):
median position error 0.56 m visible / 0.91 m blind (11.7 % relative, flat
0–25 m), fly-through-axis 15.9°, center 24.6 px, visibility accuracy 92 %.

---

## Runtime budget — the loop must run at 90–120 Hz

The deployment perception loop targets **90–120 Hz** (8.3–11 ms/frame
end-to-end). Measured: `model.step()` at batch 1, 320×192 runs **~2.4 ms
(≈425 Hz) eager fp32 on an RTX A6000**; on Jetson-class flight hardware,
TensorRT fp16 typically lands the same graph in the 4–8 ms range — inside
budget with margin for capture/decode. Export the stateful single-step graph
(hidden state as explicit I/O) with:

```bash
uv run python scripts/export_gatepose.py --checkpoint runs/gateposenet_traj/best.pt
trtexec --onnx=runs/gateposenet_traj/gateposenet_step.onnx --fp16 --saveEngine=gatepose.plan
```

Rate-awareness: the ego input carries the frame interval **dt** as its 7th
component (always known at deployment, survives ego dropout), so one model
handles 30 Hz sim streams and 90–120 Hz hardware streams — generate training
data at the deployment rate (`--traj-fps`).

## Testing on your own images

`data/test_images/` is an empty drop-in directory. Put stills or a frame
sequence there and run `scripts/run_test_images.py` — it runs the temporal
model over the files in sorted order (or `--independent` for unrelated
stills), writes annotated images plus `predictions.json` (corners, center,
metric position, rotation, visibility — original-resolution pixels, camera
optical frame). Gate constants come from `configs/gate_aigp.yaml`.

## Sources

- **MonoRace / GateNet + QuAdGate + PnP, EKF fusion, augmentation recipe,
  partial-gate PnP, extrinsics self-calibration** — *MonoRace: Winning the
  Autonomous Drone Racing Championship with a Monocular Camera*, TU Delft
  MAVLab, arXiv:2601.15222. (Basis for `gatenet/`, `perception/`, the HSV /
  brightness-gradient / affine / motion-blur augmentations, and the
  corners→PnP output path.)
- **SkyDreamer — mask-based world-model racing, GRU latent through blind
  moments, privileged-state decoding, rolling-shutter affine model, mask
  degradation** — arXiv:2510.14783. (Basis for the ConvGRU temporal core,
  blind-frame supervision, the aux self-calibration decode, and
  `synthetic/mask_degrade.py` + rolling-shutter op.)
- **Continuous 6-D rotation representation** — Zhou et al., *On the
  Continuity of Rotation Representations in Neural Networks*, CVPR 2019.
  (GatePoseNet rotation head.)
- **Domain randomization** — Tobin et al., *Domain Randomization for
  Transferring Deep Neural Networks from Simulation to the Real World*, IROS
  2017. (Randomize within true uncertainty; the calibration-jitter argument.)
- **Symmetric-object pose ambiguity** — e.g. Hodaň et al., BOP benchmark
  conventions for pose error under object symmetries. (D4 symmetry-aware
  loss/metrics.)
- **Gate/camera constants** — AI Grand Prix Virtual Qualifier Technical
  Specification VADR-TS-002, Issue 00.02 (2026-05-08); A2RL×DCL 2025 track
  configuration (stacked split-S double-gates) per the MonoRace paper.
