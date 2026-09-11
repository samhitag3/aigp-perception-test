# deploy/ — TensorRT / ONNX deployment for GatePoseNet (+ MG)

Everything needed to take a trained checkpoint to a 90–120 Hz perception loop
on the flight computer:

| file | what |
|---|---|
| `export.py` | unified ONNX exporter for **both** models (single-target GatePoseNet and multi-gate GatePoseNet-MG, auto-detected from the checkpoint); writes `<name>.onnx` + `<name>.manifest.json` |
| `trt_runtime.py` | `GatePoseTRT` (TensorRT engine, pre-allocated buffers, on-GPU hidden state) and `GatePoseORT` (onnxruntime fallback, CUDA or CPU EP) with **identical** `step(bgr_frame, ego7) -> dict` interfaces; `make_runtime()` factory; `build_engine()` (python equivalent of `trtexec --fp16`) |
| `benchmark.py` | step latency / Hz measurement (mean/p50/p95/max over 300 iters) on whichever runtime is available |

The exported graph is the **stateful single step**: `step(image (1,3,192,320),
ego (1,7), h_in (1,128,12,20)) -> outputs…, h_out`. Batch 1, static shapes,
opset 17. The runtime feeds `h_out` back as `h_in` each frame (on TensorRT the
state never leaves the GPU); call `reset()` whenever the temporal stream
breaks. Exact I/O names/shapes/dtypes, preprocessing, ego layout and frame
conventions are in the manifest JSON written next to the ONNX — the runtimes
read it at load time, so keep the two files together (an engine built as
`<stem>.plan` finds `<stem>.manifest.json` automatically).

## Quick start (this repo's dev box)

```bash
# 1. export (auto-detects single vs MG from the checkpoint)
uv run python -m deploy.export --checkpoint runs/gateposenet_traj/best.pt
# -> runs/gateposenet_traj/gateposenet_step.onnx + gateposenet_step.manifest.json

# 2. run it (TRT if installed, else onnxruntime CUDA/CPU — same interface)
uv run python -m deploy.benchmark --model runs/gateposenet_traj/gateposenet_step.onnx
```

```python
from deploy import make_runtime
rt = make_runtime("runs/gateposenet_traj/gateposenet_step.onnx")
rt.reset()
out = rt.step(bgr_frame, ego7=[vx, vy, vz, wx, wy, wz, dt])
# out["position"] (3,) m, out["R"] (3,3), out["corners_uv"] (4,2) normalized,
# out["visible_prob"], out["depth_m"], ...   (MG: leading Q dim + presence/target)
```

`ego7` is `[v_cam(3) m/s, ω_cam(3) rad/s, dt s]` in the **camera optical
frame** (+X right, +Y down, +Z forward). Pass `ego7=None` to run ego-blind
(trained with 25 % ego dropout) — but **dt must always be real**: it is the
rate-awareness input that lets one model serve 30 Hz sim and 90–120 Hz
hardware streams.

## Jetson setup (Orin NX 16 GB assumed)

**JetPack version: use JetPack 6.x** (L4T r36) → CUDA 12.x + **TensorRT
10.x**, which matches the TRT 10 tensor API this runtime targets (a
TRT 8.x/JetPack 5 fallback via `execute_async_v2` is included but untested).
Check what you have:

```bash
cat /etc/nv_tegra_release          # r36.x = JetPack 6
dpkg -l | grep -E 'nvidia-jetpack|tensorrt'
```

Install (TensorRT ships **with JetPack** — do not pip-install `tensorrt` on
Jetson):

```bash
# full JetPack meta-package (CUDA, cuDNN, TensorRT, trtexec):
sudo apt update && sudo apt install nvidia-jetpack
# or flash + install via NVIDIA sdkmanager from an x86 host.

# python TensorRT bindings (usually already present under JetPack 6):
sudo apt install python3-libnvinfer python3-libnvinfer-dev
python3 -c "import tensorrt; print(tensorrt.__version__)"   # expect 10.x

# CUDA memory bindings for trt_runtime.py — either of:
python3 -m pip install "cuda-python<13"    # preferred (match CUDA 12.x)
python3 -m pip install pycuda              # fallback (compiles ~minutes)

# runtime deps of this package:
python3 -m pip install numpy opencv-python

# optional fallback EP (Jetson wheels are NOT on PyPI — use the Jetson zoo):
#   https://elinux.org/Jetson_Zoo#ONNX_Runtime  (onnxruntime-gpu aarch64 wheel
#   matching your JetPack/Python), or CPU-only: pip install onnxruntime
```

Copy to the Jetson: the `deploy/` directory, the exported `*.onnx`, and its
`*.manifest.json`. No torch, no training code needed.

### Build the engine on the Jetson

Engines are **not portable** across GPUs/TensorRT versions — always build on
the device that will run them:

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=gateposenet_step.onnx --fp16 \
    --saveEngine=gateposenet_step.plan
# shapes are static (batch 1) so no --shapes flag is needed; to be explicit:
#   --shapes=image:1x3x192x320,ego:1x7,h_in:1x128x12x20
# python alternative (same result):
python3 -m deploy.trt_runtime --build gateposenet_step.onnx
```

Keep the `.plan` next to the `.onnx`/manifest (same stem) so
`make_runtime("gateposenet_step.plan")` finds the manifest.

Jetson power/clock state dominates latency — before benchmarking:

```bash
sudo nvpmodel -m 0          # MAXN power mode
sudo jetson_clocks           # lock clocks
```

### Measure latency

```bash
# GPU-only engine latency (TensorRT's own harness):
/usr/src/tensorrt/bin/trtexec --loadEngine=gateposenet_step.plan \
    --iterations=300 --avgRuns=100 --useSpinWait
# full python step (preprocess + inference + state feedback + postprocess):
python3 -m deploy.benchmark --model gateposenet_step.plan
```

Budget: 90–120 Hz → **8.3–11.1 ms** per step end-to-end. Reference points:
`model.step()` runs ~2.4 ms eager fp32 on an RTX A6000; TensorRT fp16 on
Orin-class hardware typically lands this graph in the 4–8 ms range — inside
budget with margin for capture/decode. Measure, don't assume.

## x86 dev box (this repo)

```bash
uv pip install onnxruntime            # CPU EP; 1.23.x has cp310 wheels
uv pip install onnxruntime-gpu        # CUDA EP (needs CUDA 12 + cuDNN 9 libs;
                                      #  see note below)
uv pip install "tensorrt-cu12<11" "cuda-python<13"
#   plain `pip install tensorrt` resolves to tensorrt-cu13 (needs a CUDA 13
#   driver) — pin the -cu12 variant on CUDA 12.x driver boxes.
```

Note (CUDA EP on the dev box): onnxruntime-gpu dlopens cuDNN 9/cuBLAS; if they
are not system-installed, point it at pip-installed NVIDIA wheels, e.g.
`LD_LIBRARY_PATH=$(.venv/bin/python -c "import os,nvidia; print(os.path.join(list(nvidia.__path__)[0]))")/cudnn/lib:...`
or simply benchmark with the TensorRT runtime / CPU EP. The ORT path exists
for interface portability, not peak speed.

## If latency is hot — model-trim knobs

In order of preference (all need a re-train, then re-export):

1. **MG decoder width/depth** (`configs/gateposenet_mg.yaml`): `d_model: 96`
   and `n_dec_layers: 1` cut the query-decoder cost roughly in half with
   modest accuracy cost. `n_queries: 6` trims further if the track never
   shows more gates.
2. **Encoder width** (`model.width_factor`): 2 → 1.5 shrinks every conv and
   the GRU state (h_in becomes `1x96x12x20` — shapes come from the manifest,
   nothing else to change).
3. **Input resolution** (`data.height/width`): 192×320 → 160×256 (keep /16
   divisible). Cheapest large win, costs small-gate range.
4. Engine-side, free: make sure the build used `--fp16`; try
   `--builderOptimizationLevel=5`; on Orin consider DLA for the conv encoder
   (`--useDLACore=0 --allowGPUFallback`) to offload the GPU — measure, DLA
   round-trips can lose.

## Output contract (fixed ordering)

Documented exhaustively in `export.py`'s docstring and machine-readably in
each `*.manifest.json`. Summary — single-target: `corners_uv,
corner_inside_logit, center_uv, position, log_depth, rot6d, visible_logit,
visible_frac, k_scale, seg_logit, h_out`; MG (per-query leading dim Q=8):
`presence_logit, target_logit, corners_uv, corner_inside_logit, center_uv,
position, log_depth, rot6d, visible_frac, mask_logit, h_out`. All poses are
**camera-optical-frame** (tilt-invariant); `R = gram_schmidt(rot6d)` and
sigmoids of the logits are computed host-side by the runtime and added to the
`step()` dict (`R`, `depth_m`, `visible_prob`/`presence_prob`/`target_prob`,
`corner_inside_prob`).
