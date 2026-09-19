"""Deployment package for GatePoseNet / GatePoseNet-MG.

ONNX export + TensorRT / onnxruntime inference for the 90-120 Hz stateful
perception step ``step(image, ego, h) -> outputs + h``.

    from deploy import make_runtime
    rt = make_runtime("gateposenet_step.plan")   # or .onnx
    rt.reset()
    out = rt.step(bgr_frame, ego7)               # dict, camera optical frame

See deploy/README.md for Jetson setup, engine building and latency
measurement. ``deploy.export`` produces the ONNX graph + JSON manifest;
``deploy.benchmark`` measures step latency on whatever runtime is available.

This package deliberately depends only on numpy + opencv at inference time
(torch is needed for export only), so it can be copied to the flight computer
as-is.
"""

from deploy.trt_runtime import (  # noqa: F401
    GatePoseORT,
    GatePoseTRT,
    build_engine,
    make_runtime,
    preprocess_bgr,
)

__all__ = [
    "GatePoseORT",
    "GatePoseTRT",
    "build_engine",
    "make_runtime",
    "preprocess_bgr",
]
