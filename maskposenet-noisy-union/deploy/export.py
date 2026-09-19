#!/usr/bin/env python3
"""Unified ONNX exporter for GatePoseNet (single-target) and GatePoseNet-MG.

Exports the STATEFUL single-step deployment graph — the recurrent ConvGRU
state is an explicit input/output tensor (``h_in`` / ``h_out``), so the
runtime feeds it back each frame and the 90-120 Hz loop needs exactly one
graph execution per frame:

    step(image (1,3,H,W), ego (1,7), h_in (1,C5,H/16,W/16))
        -> named outputs ... , h_out (same shape as h_in)

Model type is AUTO-DETECTED from the checkpoint: MG checkpoints contain the
query-decoder weights (``queries.weight``); the training config saved inside
the checkpoint (``extra.config``) is used unless ``--config`` overrides it.

Usage:
    uv run python -m deploy.export --checkpoint runs/gateposenet_traj/best.pt
    uv run python -m deploy.export --checkpoint runs/gateposenet_mg/best.pt \
        --out deploy_out/gateposenet_mg_step.onnx

Writes ``<name>.onnx`` plus ``<name>.manifest.json`` describing every
input/output (name, shape, dtype, semantics), the expected preprocessing,
ego-vector layout / dt semantics, and the camera-frame conventions. The
manifest is what deploy/trt_runtime.py reads at load time — keep the two
files next to each other (an engine built as ``<name>.plan`` from
``<name>.onnx`` picks up the same manifest by stem).

FIXED OUTPUT ORDERING (position in the ONNX output list is part of the
deployment contract; do not reorder without re-exporting everywhere):

  single-target (GatePoseNet):
      0 corners_uv           (1,4,2)   image-normalized [0..1] x (u,v), TL,TR,BR,BL
      1 corner_inside_logit  (1,4)     per-corner in-frame logits
      2 center_uv            (1,2)     fly-through center, image-normalized
      3 position             (1,3)     gate center, meters, camera optical frame
      4 log_depth            (1,)      log(gate center depth in meters)
      5 rot6d                (1,6)     Zhou 6-D rotation -> R_cam_gate (Gram-Schmidt)
      6 visible_logit        (1,)      gate-in-frame confidence logit
      7 visible_frac         (1,)      visible fraction, already sigmoided
      8 k_scale              (1,2)     (fx,fy)/nominal self-calibration
      9 seg_logit            (1,1,H/4,W/4)  aux gate mask logits
     10 h_out                (1,C5,H/16,W/16) recurrent state -> next h_in

  multi-gate (GatePoseNetMG), Q = n_queries:
      0 presence_logit       (1,Q)     per-query "this query is a gate" logit
      1 target_logit         (1,Q)     per-query "the gate I am flying at" logit
      2 corners_uv           (1,Q,4,2)
      3 corner_inside_logit  (1,Q,4)
      4 center_uv            (1,Q,2)
      5 position             (1,Q,3)
      6 log_depth            (1,Q)
      7 rot6d                (1,Q,6)
      8 visible_frac         (1,Q)
      9 mask_logit           (1,Q,H/4,W/4)  per-query instance mask logits
     10 h_out                (1,C5,H/16,W/16)

``R`` (the 3x3 rotation) is intentionally NOT exported — it is derived from
rot6d and the runtime recomputes it host-side (deploy.trt_runtime), keeping
the graph free of the Gram-Schmidt ops TensorRT gains nothing from.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatenet.utils import load_config, pick_device  # noqa: E402
from gateposenet.model import build_gateposenet_mg  # noqa: E402
from gateposenet.model_single import \
    build_gateposenet_single as build_gateposenet  # noqa: E402

OPSET = 18

SINGLE_OUTPUTS = ("corners_uv", "corner_inside_logit", "center_uv",
                  "position", "log_depth", "rot6d", "visible_logit",
                  "visible_frac", "k_scale", "seg_logit")
MG_OUTPUTS = ("presence_logit", "target_logit", "corners_uv",
              "corner_inside_logit", "center_uv", "position", "log_depth",
              "rot6d", "visible_frac", "mask_logit")

OUTPUT_DESC = {
    "corners_uv": "gate corners (TL,TR,BR,BL), image-normalized [0..1] (u,v) "
                  "— multiply by original frame WxH for pixels",
    "corner_inside_logit": "per-corner in-frame logits (sigmoid -> prob)",
    "center_uv": "fly-through center, image-normalized (u,v)",
    "position": "gate center in meters, camera OPTICAL frame "
                "(+X right, +Y down, +Z forward)",
    "log_depth": "log(gate center depth in meters); depth_m = exp(log_depth)",
    "rot6d": "Zhou 6-D rotation; Gram-Schmidt -> R_cam_gate, "
             "fly-through axis = R[:,2]",
    "visible_logit": "gate-in-frame confidence logit (sigmoid -> prob)",
    "visible_frac": "predicted visible fraction of the gate (already 0..1)",
    "k_scale": "(fx, fy) relative to nominal — aux self-calibration",
    "seg_logit": "aux gate-mask logits at 1/4 input resolution",
    "presence_logit": "per-query gate-presence logit (sigmoid -> prob)",
    "target_logit": "per-query 'the gate I am flying at' logit",
    "mask_logit": "per-query instance-mask logits at 1/4 input resolution",
}

DTYPE_NAME = {torch.float32: "float32", torch.float16: "float16"}


class _StepGraphSingle(torch.nn.Module):
    """ONNX wrapper for the single recurrent-state variant."""

    def __init__(self, net, output_keys):
        super().__init__()
        self.net = net
        self.output_keys = tuple(output_keys)

    def forward(self, image, ego, h_in):
        out, h_out = self.net.step(image, ego, h_in)
        return tuple(out[k] for k in self.output_keys) + (h_out,)


class _StepGraphDual(torch.nn.Module):
    """ONNX wrapper for temporal_highres=True (h5 + h4 recurrent states)."""

    def __init__(self, net, output_keys):
        super().__init__()
        self.net = net
        self.output_keys = tuple(output_keys)

    def forward(self, image, ego, h5_in, h4_in):
        out, hidden = self.net.step(image, ego, (h5_in, h4_in))
        h5_out, h4_out = hidden
        return tuple(out[k] for k in self.output_keys) + (h5_out, h4_out)


def detect_model_type(state_dict: dict) -> str:
    return "mg" if "queries.weight" in state_dict else "single"


def load_net(checkpoint: str | Path, config: str | None = None,
             device: torch.device | str = "cpu"):
    """Build + load either model from a checkpoint. Returns (net, type, cfg)."""
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)
    model_type = detect_model_type(sd)
    if config:
        cfg = load_config(config)
    else:
        cfg = (ckpt.get("extra") or {}).get("config")
        if cfg is None:
            default = ("configs/gateposenet_mg.yaml" if model_type == "mg"
                       else "configs/gateposenet_traj.yaml")
            print(f"warning: checkpoint has no embedded config — "
                  f"falling back to {default}")
            cfg = load_config(default)
    build = build_gateposenet_mg if model_type == "mg" else build_gateposenet
    net = build(cfg["model"]).to(device)
    net.load_state_dict(sd, strict=True)
    net.eval()
    return net, model_type, cfg


def export(checkpoint: str, out: str | None = None, config: str | None = None,
           device: str | None = None, parity: bool = True) -> Path:
    dev = pick_device(device)
    net, model_type, cfg = load_net(checkpoint, config, dev)
    d = cfg.get("data", {})
    h_img, w_img = int(d.get("height", 192)), int(d.get("width", 320))
    output_keys = MG_OUTPUTS if model_type == "mg" else SINGLE_OUTPUTS

    image = torch.rand(1, 3, h_img, w_img, device=dev)
    ego = torch.zeros(1, 7, device=dev)
    ego[0, 6] = 1.0 / 120.0
    hidden = net.init_hidden(1, (h_img, w_img), device=dev)

    temporal_highres = isinstance(hidden, (tuple, list))
    if temporal_highres:
        h5, h4 = hidden
        wrapper = _StepGraphDual(net, output_keys).eval()
        export_inputs = (image, ego, h5, h4)
        input_names = ["image", "ego", "h5_in", "h4_in"]
        state_output_names = ["h5_out", "h4_out"]
    else:
        wrapper = _StepGraphSingle(net, output_keys).eval()
        export_inputs = (image, ego, hidden)
        input_names = ["image", "ego", "h_in"]
        state_output_names = ["h_out"]

    if out is None:
        out = str(Path(checkpoint).parent
                  / f"gateposenet_{'mg_' if model_type == 'mg' else ''}step.onnx")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_names = list(output_keys) + state_output_names
    torch.onnx.export(
        wrapper, export_inputs, str(out_path),
        input_names=input_names,
        output_names=all_names,
        opset_version=OPSET, do_constant_folding=True,
    )

    # Trace once to record exact output shapes for the manifest.
    with torch.no_grad():
        ref = wrapper(*export_inputs)
    manifest = _build_manifest(
        model_type, cfg, checkpoint, out_path,
        export_inputs, input_names, all_names, ref,
    )
    man_path = out_path.parent / (out_path.stem + ".manifest.json")
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")
    n_par = sum(p.numel() for p in net.parameters())
    print(f"exported {model_type} model ({n_par / 1e6:.2f} M params): "
          f"{out_path} ({out_path.stat().st_size / 1e6:.1f} MB, opset {OPSET})")
    print(f"manifest: {man_path}")

    if parity:
        _parity_check(wrapper, export_inputs, out_path)
    return out_path


def _tensor_spec(name: str, t: torch.Tensor, desc: str) -> dict:
    return {"name": name, "shape": list(t.shape),
            "dtype": DTYPE_NAME.get(t.dtype, str(t.dtype)), "desc": desc}


def _build_manifest(model_type, cfg, checkpoint, out_path, inputs,
                    input_names, out_names, ref_outputs) -> dict:
    image = inputs[0]
    outputs = [_tensor_spec(n, t, OUTPUT_DESC.get(n, ""))
               for n, t in zip(out_names, ref_outputs)]

    input_specs = [
        _tensor_spec(
            "image", image,
            "binary UNION gate mask repeated across 3 channels; float32 in "
            "[0,1], resized to "
            f"{image.shape[3]}x{image.shape[2]} with INTER_NEAREST"),
        _tensor_spec(
            "ego", inputs[1],
            "[vx, vy, vz, wx, wy, wz, dt] — linear (m/s) and angular "
            "(rad/s) velocity in the CAMERA OPTICAL frame, plus dt (s)"),
    ]

    if len(inputs) == 4:
        input_specs.extend([
            _tensor_spec("h5_in", inputs[2],
                         "low-resolution ConvGRU recurrent state; zeros at sequence start"),
            _tensor_spec("h4_in", inputs[3],
                         "high-resolution ConvGRU recurrent state; zeros at sequence start"),
        ])
        state = {
            "inputs": ["h5_in", "h4_in"],
            "outputs": ["h5_out", "h4_out"],
            "init": "zeros",
            "note": "feed both recurrent outputs back as inputs on the next frame; reset both when the temporal stream breaks",
        }
    else:
        input_specs.append(_tensor_spec(
            "h_in", inputs[2], "ConvGRU recurrent state; zeros at sequence start"))
        state = {
            "input": "h_in", "output": "h_out", "init": "zeros",
            "note": "feed recurrent output back as input on the next frame; reset when the temporal stream breaks",
        }

    infer_cfg = cfg.get("inference", {})
    man = {
        "model_type": model_type,
        "onnx_file": out_path.name,
        "opset": OPSET,
        "checkpoint": str(checkpoint),
        "model_config": cfg.get("model", {}),
        "input_size": {"height": image.shape[2], "width": image.shape[3]},
        "inputs": input_specs,
        "outputs": outputs,
        "state": state,
        "preprocessing": {
            "source": "mask only; RGB is visualization-only and is NOT a model input",
            "unionize": "all non-zero instance-mask pixels -> foreground 1",
            "channels": "repeat the binary union mask across 3 channels",
            "resize": f"{image.shape[3]}x{image.shape[2]}, cv2.INTER_NEAREST",
            "scale": "0/1 float32",
            "mean": None, "std": None,
            "layout": "NCHW, float32, batch 1",
        },
        "inference": {
            "presence_threshold": float(infer_cfg.get("presence_threshold", 0.5)),
            "mask_threshold": float(infer_cfg.get("mask_threshold", 0.5)),
            "note": "thresholds are host-side postprocessing; presence probability = sigmoid(presence_logit)",
        },
        "ego_semantics": {
            "layout": ["vx", "vy", "vz", "wx", "wy", "wz", "dt"],
            "frame": "camera optical (OpenCV: +X right, +Y down, +Z forward)",
            "units": ["m/s"] * 3 + ["rad/s"] * 3 + ["s"],
            "dt": "frame interval in seconds",
        },
        "frames": {
            "camera_optical": "OpenCV: +X right, +Y down, +Z forward",
            "gate_body": "+X right, +Y down, +Z = fly-through axis pointing away from the approach side",
            "outputs_are_camera_frame": True,
        },
        "gate": {"outer_m": 2.7, "inner_m": 1.5, "symmetry": "D4"},
        "normalization_2d": "corners_uv / center_uv are normalized by image size; multiply by original frame WxH for pixels",
        "derived_host_side": {
            "R_cam_gate": "Gram-Schmidt on rot6d",
            "depth_m": "exp(log_depth)",
        },
    }
    if model_type == "mg":
        man["n_queries"] = int(cfg["model"].get("n_queries", 8))
        man["mg_note"] = (
            "per-query outputs; select gates by sigmoid(presence_logit) using "
            "the tuned host-side threshold; flown target by argmax(target_logit)")
    return man

def _parity_check(wrapper, inputs, out_path):
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed — skipped parity check")
        return

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    ort_inputs = sess.get_inputs()
    if len(ort_inputs) != len(inputs):
        raise RuntimeError(
            f"ONNX input count mismatch: graph has {len(ort_inputs)}, "
            f"PyTorch wrapper has {len(inputs)}")
    feeds = {
        meta.name: tensor.detach().cpu().numpy()
        for meta, tensor in zip(ort_inputs, inputs)
    }
    outs = sess.run(None, feeds)
    with torch.no_grad():
        ref = wrapper(*inputs)
    if len(outs) != len(ref):
        raise RuntimeError(
            f"ONNX output count mismatch: graph has {len(outs)}, "
            f"PyTorch wrapper has {len(ref)}")
    err = max(
        float(abs(o - r.detach().cpu().numpy()).max())
        for o, r in zip(outs, ref)
    )
    print(f"onnxruntime parity (CPU EP vs torch): max abs err {err:.2e}"
          + ("  [OK]" if err < 1e-3 else "  [WARN: check the export]"))

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", default="runs/gateposenet_traj/best.pt")
    ap.add_argument("--config", default=None,
                    help="override the config embedded in the checkpoint")
    ap.add_argument("--out", default=None,
                    help="output .onnx path (default: next to the checkpoint)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-parity", action="store_true",
                    help="skip the onnxruntime parity check")
    args = ap.parse_args()
    export(args.checkpoint, args.out, args.config, args.device,
           parity=not args.no_parity)


if __name__ == "__main__":
    main()
