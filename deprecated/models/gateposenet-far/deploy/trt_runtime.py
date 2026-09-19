"""GatePoseNet inference runtimes: TensorRT (flight) + onnxruntime (fallback).

Both runtimes expose the SAME interface, so downstream code is
runtime-agnostic:

    rt = make_runtime("gateposenet_step.plan")        # or .onnx
    rt.reset()                                        # zero the ConvGRU state
    out = rt.step(bgr_frame, ego7)                    # per camera frame
    # out: dict of numpy arrays (batch dim stripped) + derived keys:
    #      "R" (3x3 or Qx3x3 from rot6d), "depth_m", "visible_prob" /
    #      "presence_prob"+"target_prob" (MG)

``step`` does the whole per-frame job: resize + BGR->RGB + /255 (exactly the
training preprocessing), inference, and hidden-state feedback (h_out -> h_in;
on TensorRT the state never leaves the GPU).

Dependencies at inference time: numpy + opencv, plus
  * TensorRT path: ``tensorrt`` + (``cuda-python`` or ``pycuda``)
  * fallback path: ``onnxruntime`` (CUDA or CPU EP — runs anywhere)
No torch. See deploy/README.md for install instructions per platform.

Engine building:  python -m deploy.trt_runtime --build model.onnx  (or trtexec)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

ENGINE_SUFFIXES = {".plan", ".engine", ".trt"}


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def preprocess_bgr(frame: np.ndarray, wh: tuple[int, int]) -> np.ndarray:
    """BGR uint8 (any size) -> (1,3,H,W) float32 RGB in [0,1].

    Matches training exactly: cv2.resize INTER_AREA, BGR->RGB, /255,
    no mean/std normalization.
    """
    w, h = wh
    if frame.shape[1] != w or frame.shape[0] != h:
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x = rgb.transpose(2, 0, 1)[None].astype(np.float32)
    x /= 255.0
    return np.ascontiguousarray(x)


def rot6d_to_matrix_np(x: np.ndarray) -> np.ndarray:
    """(...,6) Zhou 6-D rotation -> (...,3,3) via Gram-Schmidt (numpy)."""
    a1, a2 = x[..., 0:3], x[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2p = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2p / np.linalg.norm(a2p, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def find_manifest(model_path: str | Path) -> Path | None:
    """<stem>.manifest.json next to the model (engine shares the onnx stem)."""
    p = Path(model_path)
    cand = p.parent / (p.stem + ".manifest.json")
    return cand if cand.exists() else None


class _RuntimeBase:
    """Preprocessing + state feedback + postprocessing shared by ORT/TRT.

    Subclasses implement ``_infer(image, ego) -> list[np.ndarray]`` returning
    the outputs in graph order WITHOUT h_out (state feedback is theirs to
    handle — ORT round-trips it through host numpy, TRT keeps it on-device).
    """

    def __init__(self, model_path: str | Path, manifest: str | Path | None):
        self.model_path = Path(model_path)
        man_path = Path(manifest) if manifest else find_manifest(model_path)
        if man_path is None:
            raise FileNotFoundError(
                f"no manifest found for {model_path} — expected "
                f"{Path(model_path).stem}.manifest.json next to it "
                f"(deploy/export.py writes it)")
        self.manifest = json.loads(Path(man_path).read_text())
        self.model_type = self.manifest["model_type"]
        s = self.manifest["input_size"]
        self.input_wh = (int(s["width"]), int(s["height"]))
        self.h_shape = tuple(self.manifest["inputs"][2]["shape"])
        # output names in graph order, h_out last by the export contract
        self.output_names = [o["name"] for o in self.manifest["outputs"]]
        assert self.output_names[-1] == "h_out", "manifest violates contract"
        self.head_names = self.output_names[:-1]
        self.default_dt = 1.0 / 120.0

    # -- state ------------------------------------------------------------ #
    def reset(self) -> None:
        """Zero the recurrent state (call at sequence start / stream break)."""
        raise NotImplementedError

    def _infer(self, image: np.ndarray, ego: np.ndarray) -> list[np.ndarray]:
        raise NotImplementedError

    # -- the 90-120 Hz API -------------------------------------------------- #
    def step(self, bgr_frame: np.ndarray, ego7=None,
             dt: float | None = None) -> dict:
        """One perception step: BGR frame (+ ego) -> output dict.

        ego7: [vx,vy,vz,wx,wy,wz,dt] in the camera optical frame; pass None
        to run ego-blind (zeros — valid, the model trains with ego dropout),
        in which case ``dt`` (or the default 1/120 s) still fills slot 6.
        """
        image = preprocess_bgr(bgr_frame, self.input_wh)
        if ego7 is None:
            ego = np.zeros((1, 7), np.float32)
            ego[0, 6] = self.default_dt if dt is None else float(dt)
        else:
            ego = np.asarray(ego7, np.float32).reshape(1, 7).copy()
            if dt is not None:
                ego[0, 6] = float(dt)
        outs = self._infer(image, ego)
        return self._postprocess(outs)

    def _postprocess(self, outs: list[np.ndarray]) -> dict:
        d = {n: o[0] for n, o in zip(self.head_names, outs)}  # strip batch
        d["R"] = rot6d_to_matrix_np(d["rot6d"])
        d["depth_m"] = np.exp(d["log_depth"])
        if self.model_type == "single":
            d["visible_prob"] = _sigmoid(d["visible_logit"])
        else:
            d["presence_prob"] = _sigmoid(d["presence_logit"])
            d["target_prob"] = _sigmoid(d["target_logit"])
        d["corner_inside_prob"] = _sigmoid(d["corner_inside_logit"])
        return d


# --------------------------------------------------------------------------- #
# onnxruntime fallback — runs anywhere (CUDA EP if present, else CPU)
# --------------------------------------------------------------------------- #
def _preload_cuda_ep_libs() -> None:
    """Best-effort: dlopen cuDNN/cuBLAS from pip 'nvidia-*' wheels so ORT's
    CUDA EP finds them without LD_LIBRARY_PATH (torch's bundled copies work
    too — importing torch first has the same effect)."""
    import ctypes
    import glob
    try:
        import nvidia
        roots = list(nvidia.__path__)
    except ImportError:
        return
    # ORT's CUDA EP dlopens all of these; load order respects dependencies
    for sub in ("cuda_runtime", "cuda_nvrtc", "cublas", "cufft", "curand",
                "cudnn"):
        for root in roots:
            for so in sorted(glob.glob(f"{root}/{sub}/lib/lib*.so.*")):
                try:
                    ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


class GatePoseORT(_RuntimeBase):
    def __init__(self, onnx_path: str | Path,
                 manifest: str | Path | None = None, providers=None):
        super().__init__(onnx_path, manifest)
        import onnxruntime as ort
        if providers is None and \
                "CUDAExecutionProvider" in ort.get_available_providers():
            _preload_cuda_ep_libs()
        if providers is None:
            avail = ort.get_available_providers()
            providers = []
            if "CUDAExecutionProvider" in avail:
                # tf32 accumulates visible error through the recurrent state
                # (~2e-2 after 20 steps); off costs little on this small net
                providers.append(("CUDAExecutionProvider", {"use_tf32": "0"}))
            providers.append("CPUExecutionProvider")
        so = ort.SessionOptions()
        so.graph_optimization_level = \
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(onnx_path), sess_options=so,
                                         providers=providers)
        self.provider = self.sess.get_providers()[0]
        self._out_order = list(self.output_names)  # ask ORT in contract order
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros(self.h_shape, np.float32)

    def _infer(self, image: np.ndarray, ego: np.ndarray) -> list[np.ndarray]:
        outs = self.sess.run(self._out_order,
                             {"image": image, "ego": ego, "h_in": self._h})
        self._h = outs[-1]
        return outs[:-1]


# --------------------------------------------------------------------------- #
# CUDA memory backend for TensorRT (cuda-python preferred, pycuda fallback)
# --------------------------------------------------------------------------- #
class _Cuda:
    """Minimal device-memory + stream wrapper over cuda-python or pycuda."""

    def __init__(self):
        self.kind = None
        try:  # cuda-python >= 13 moved the bindings
            try:
                from cuda.bindings import runtime as cudart
            except ImportError:
                from cuda import cudart
            self._rt = cudart
            err, self.stream = cudart.cudaStreamCreate()
            self._ck(err)
            self.kind = "cuda-python"
            return
        except ImportError:
            pass
        try:
            import pycuda.autoinit  # noqa: F401  (creates the context)
            import pycuda.driver as drv
            self._drv = drv
            self.stream = drv.Stream()
            self.kind = "pycuda"
            return
        except ImportError:
            raise ImportError(
                "TensorRT runtime needs a CUDA binding: "
                "pip install cuda-python  (or pycuda)")

    def _ck(self, err):
        if int(err) != 0:
            raise RuntimeError(f"CUDA error {err}")

    # cuda-python passes raw stream handles; pycuda wraps them
    @property
    def stream_handle(self) -> int:
        if self.kind == "cuda-python":
            return int(self.stream)
        return self.stream.handle

    def alloc(self, nbytes: int) -> int:
        if self.kind == "cuda-python":
            err, ptr = self._rt.cudaMalloc(nbytes)
            self._ck(err)
            return int(ptr)
        mem = self._drv.mem_alloc(nbytes)
        # keep a reference — pycuda frees the allocation when it is GC'd
        self._allocs = getattr(self, "_allocs", [])
        self._allocs.append(mem)
        return int(mem)

    def h2d(self, dst_ptr: int, src: np.ndarray) -> None:
        if self.kind == "cuda-python":
            k = self._rt.cudaMemcpyKind.cudaMemcpyHostToDevice
            self._ck(self._rt.cudaMemcpyAsync(dst_ptr, src.ctypes.data,
                                              src.nbytes, k, self.stream)[0])
        else:
            self._drv.memcpy_htod_async(dst_ptr, src, self.stream)

    def d2h(self, dst: np.ndarray, src_ptr: int) -> None:
        if self.kind == "cuda-python":
            k = self._rt.cudaMemcpyKind.cudaMemcpyDeviceToHost
            self._ck(self._rt.cudaMemcpyAsync(dst.ctypes.data, src_ptr,
                                              dst.nbytes, k, self.stream)[0])
        else:
            self._drv.memcpy_dtoh_async(dst, src_ptr, self.stream)

    def d2d(self, dst_ptr: int, src_ptr: int, nbytes: int) -> None:
        if self.kind == "cuda-python":
            k = self._rt.cudaMemcpyKind.cudaMemcpyDeviceToDevice
            self._ck(self._rt.cudaMemcpyAsync(dst_ptr, src_ptr, nbytes, k,
                                              self.stream)[0])
        else:
            self._drv.memcpy_dtod_async(dst_ptr, src_ptr, nbytes, self.stream)

    def memset0(self, ptr: int, nbytes: int) -> None:
        if self.kind == "cuda-python":
            self._ck(self._rt.cudaMemsetAsync(ptr, 0, nbytes, self.stream)[0])
        else:
            self._drv.memset_d8_async(ptr, 0, nbytes, self.stream)

    def sync(self) -> None:
        if self.kind == "cuda-python":
            self._ck(self._rt.cudaStreamSynchronize(self.stream)[0])
        else:
            self.stream.synchronize()


# --------------------------------------------------------------------------- #
# TensorRT runtime — the flight path
# --------------------------------------------------------------------------- #
class GatePoseTRT(_RuntimeBase):
    """TensorRT engine runner with pre-allocated buffers and on-GPU state.

    Supports the TensorRT 10.x tensor API (execute_async_v3, JetPack 6) and
    falls back to the 8.x binding API (execute_async_v2, JetPack 5).
    """

    def __init__(self, engine_path: str | Path,
                 manifest: str | Path | None = None):
        super().__init__(engine_path, manifest)
        import tensorrt as trt
        self.trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        with open(engine_path, "rb") as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"failed to deserialize engine {engine_path}")
        self.engine = engine
        self.context = engine.create_execution_context()
        self.cuda = _Cuda()
        self._new_api = hasattr(engine, "num_io_tensors")

        # pre-allocate one device buffer + pinned-ish host array per tensor
        self._dev: dict[str, int] = {}
        self._host: dict[str, np.ndarray] = {}
        self._nbytes: dict[str, int] = {}
        for name, shape in self._io_tensors():
            arr = np.zeros(shape, np.float32)
            self._dev[name] = self.cuda.alloc(arr.nbytes)
            self._host[name] = arr
            self._nbytes[name] = arr.nbytes
            if self._new_api:
                self.context.set_tensor_address(name, self._dev[name])
        if not self._new_api:
            self._bindings = [self._dev[self.engine.get_binding_name(i)]
                              for i in range(self.engine.num_bindings)]
        self.reset()

    def _io_tensors(self):
        if self._new_api:
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                yield name, tuple(self.engine.get_tensor_shape(name))
        else:
            for i in range(self.engine.num_bindings):
                yield (self.engine.get_binding_name(i),
                       tuple(self.engine.get_binding_shape(i)))

    def reset(self) -> None:
        self.cuda.memset0(self._dev["h_in"], self._nbytes["h_in"])
        self.cuda.sync()

    def _infer(self, image: np.ndarray, ego: np.ndarray) -> list[np.ndarray]:
        cu = self.cuda
        cu.h2d(self._dev["image"], np.ascontiguousarray(image, np.float32))
        cu.h2d(self._dev["ego"], np.ascontiguousarray(ego, np.float32))
        if self._new_api:
            ok = self.context.execute_async_v3(cu.stream_handle)
        else:
            ok = self.context.execute_async_v2(self._bindings,
                                               cu.stream_handle)
        if not ok:
            raise RuntimeError("TensorRT execution failed")
        for name in self.head_names:
            cu.d2h(self._host[name], self._dev[name])
        # feed the state back entirely on-device
        cu.d2d(self._dev["h_in"], self._dev["h_out"], self._nbytes["h_out"])
        cu.sync()
        # copy: the host staging buffers are overwritten by the next step
        return [self._host[n].copy() for n in self.head_names]


# --------------------------------------------------------------------------- #
# engine building + factory
# --------------------------------------------------------------------------- #
def build_engine(onnx_path: str | Path, plan_path: str | Path | None = None,
                 fp16: bool = True, workspace_gb: float = 2.0) -> Path:
    """Build a TensorRT engine from an exported ONNX graph (static, batch 1).

    Equivalent to:  trtexec --onnx=<onnx> --fp16 --saveEngine=<plan>
    """
    import tensorrt as trt
    onnx_path = Path(onnx_path)
    if plan_path is None:
        plan_path = onnx_path.with_suffix(".plan")
    plan_path = Path(plan_path)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        msgs = "\n".join(str(parser.get_error(i))
                         for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed:\n{msgs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                 int(workspace_gb * (1 << 30)))
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    t0 = time.perf_counter()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("TensorRT engine build failed")
    plan_path.write_bytes(blob)
    print(f"built {plan_path} ({plan_path.stat().st_size / 1e6:.1f} MB, "
          f"fp16={fp16}) in {time.perf_counter() - t0:.1f}s")
    return plan_path


def _tensorrt_available() -> bool:
    try:
        import tensorrt  # noqa: F401
        _Cuda()
        return True
    except Exception:
        return False


def make_runtime(path: str | Path, manifest: str | Path | None = None,
                 prefer_trt: bool = True, fp16: bool = True):
    """Factory: TensorRT if available, else onnxruntime — same interface.

    * ``.plan``/``.engine``/``.trt`` path -> GatePoseTRT
    * ``.onnx`` path -> if TensorRT is importable (and prefer_trt), build (or
      reuse) ``<stem>.plan`` next to it and return GatePoseTRT; otherwise
      GatePoseORT (CUDA EP if available, else CPU).
    """
    path = Path(path)
    if path.suffix in ENGINE_SUFFIXES:
        return GatePoseTRT(path, manifest)
    if path.suffix != ".onnx":
        raise ValueError(f"expected .onnx or engine file, got {path}")
    if prefer_trt and _tensorrt_available():
        plan = path.with_suffix(".plan")
        if not plan.exists() or plan.stat().st_mtime < path.stat().st_mtime:
            print(f"building TensorRT engine {plan.name} (one-time)...")
            build_engine(path, plan, fp16=fp16)
        try:
            return GatePoseTRT(plan, manifest)
        except Exception as e:  # engine from another TRT version, etc.
            print(f"TensorRT runtime failed ({e}) — falling back to "
                  f"onnxruntime")
    return GatePoseORT(path, manifest)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Build a TensorRT engine from an exported ONNX graph")
    ap.add_argument("--build", metavar="ONNX", required=True)
    ap.add_argument("--out", default=None, help="engine path (default: "
                                                "<onnx stem>.plan)")
    ap.add_argument("--fp32", action="store_true",
                    help="disable fp16 (default: fp16 on)")
    ap.add_argument("--workspace-gb", type=float, default=2.0)
    a = ap.parse_args()
    build_engine(a.build, a.out, fp16=not a.fp32, workspace_gb=a.workspace_gb)
