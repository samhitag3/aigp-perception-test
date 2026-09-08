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

OPSET = 17

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


class _StepGraph(torch.nn.Module):
    """ONNX wrapper: fixed output order, hidden state as explicit I/O."""

    def __init__(self, net, output_keys):
        super().__init__()
        self.net = net
        self.output_keys = tuple(output_keys)

    def forward(self, image, ego, h):
        out, h2 = self.net.step(image, ego, h)
        return tuple(out[k] for k in self.output_keys) + (h2,)


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
    wrapper = _StepGraph(net, output_keys).eval()

    image = torch.rand(1, 3, h_img, w_img, device=dev)
    ego = torch.zeros(1, 7, device=dev)
    ego[0, 6] = 1.0 / 120.0
    hidden = net.init_hidden(1, (h_img, w_img), device=dev)

    if out is None:
        out = str(Path(checkpoint).parent
                  / f"gateposenet_{'mg_' if model_type == 'mg' else ''}step.onnx")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_names = list(output_keys) + ["h_out"]
    torch.onnx.export(
        wrapper, (image, ego, hidden), str(out_path),
        input_names=["image", "ego", "h_in"],
        output_names=all_names,
        opset_version=OPSET, do_constant_folding=True,
    )

    # trace once to record the exact output shapes for the manifest
    with torch.no_grad():
        ref = wrapper(image, ego, hidden)
    manifest = _build_manifest(model_type, cfg, checkpoint, out_path,
                               (image, ego, hidden), all_names, ref)
    man_path = out_path.parent / (out_path.stem + ".manifest.json")
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")
    n_par = sum(p.numel() for p in net.parameters())
    print(f"exported {model_type} model ({n_par / 1e6:.2f} M params): "
          f"{out_path} ({out_path.stat().st_size / 1e6:.1f} MB, opset {OPSET})")
    print(f"manifest: {man_path}")

    if parity:
        _parity_check(wrapper, (image, ego, hidden), out_path)
    return out_path


def _tensor_spec(name: str, t: torch.Tensor, desc: str) -> dict:
    return {"name": name, "shape": list(t.shape),
            "dtype": DTYPE_NAME.get(t.dtype, str(t.dtype)), "desc": desc}


def _build_manifest(model_type, cfg, checkpoint, out_path, inputs, out_names,
                    ref_outputs) -> dict:
    image, ego, hidden = inputs
    outputs = [_tensor_spec(n, t, OUTPUT_DESC.get(n, ""))
               for n, t in zip(out_names[:-1], ref_outputs[:-1])]
    outputs.append(_tensor_spec(
        "h_out", ref_outputs[-1],
        "ConvGRU recurrent state — feed back as h_in on the next frame"))
    man = {
        "model_type": model_type,
        "onnx_file": out_path.name,
        "opset": OPSET,
        "checkpoint": str(checkpoint),
        "model_config": cfg.get("model", {}),
        "input_size": {"height": image.shape[2], "width": image.shape[3]},
        "inputs": [
            _tensor_spec("image", image,
                         "RGB frame, float32 in [0,1] (uint8/255), resized "
                         f"to {image.shape[3]}x{image.shape[2]} (INTER_AREA), "
                         "NO mean/std normalization"),
            _tensor_spec("ego", ego,
                         "[vx, vy, vz, wx, wy, wz, dt] — linear (m/s) and "
                         "angular (rad/s) velocity in the CAMERA OPTICAL "
                         "frame, plus the frame interval dt (s)"),
            _tensor_spec("h_in", hidden,
                         "recurrent state; zeros at sequence start"),
        ],
        "outputs": outputs,
        "state": {"input": "h_in", "output": "h_out", "init": "zeros",
                  "note": "batch-1 stateful step graph; reset (zeros) when "
                          "the temporal stream breaks"},
        "preprocessing": {
            "color": "RGB (convert from BGR capture)",
            "resize": f"{image.shape[3]}x{image.shape[2]}, cv2.INTER_AREA",
            "scale": "1/255.0", "mean": None, "std": None,
            "layout": "NCHW, float32, batch 1",
        },
        "ego_semantics": {
            "layout": ["vx", "vy", "vz", "wx", "wy", "wz", "dt"],
            "frame": "camera optical (OpenCV: +X right, +Y down, +Z forward)",
            "units": ["m/s"] * 3 + ["rad/s"] * 3 + ["s"],
            "dt": "frame interval in seconds — ALWAYS set it (rate-awareness: "
                  "one model serves 30 Hz sim and 90-120 Hz hardware); "
                  "v/omega may be zeroed when no EKF estimate is available "
                  "(the model is trained with 25% ego dropout)",
        },
        "frames": {
            "camera_optical": "OpenCV: +X right, +Y down, +Z forward",
            "gate_body": "+X right, +Y down, +Z = fly-through axis pointing "
                         "away from the approach side; head-on upright view "
                         "reads R_cam_gate = identity",
            "outputs_are_camera_frame": True,
            "camera_to_body": "one fixed known rotation from the mount tilt "
                              "(sam3: synthetic.pose_gt.optical_to_body)",
        },
        "gate": {"outer_m": 2.7, "inner_m": 1.5, "symmetry": "D4"},
        "normalization_2d": "corners_uv / center_uv are normalized by IMAGE "
                            "SIZE (never by K); multiply by the original "
                            "frame WxH to get pixels",
        "derived_host_side": {
            "R_cam_gate": "Gram-Schmidt on rot6d (deploy.trt_runtime does it)",
            "depth_m": "exp(log_depth)",
        },
    }
    if model_type == "mg":
        man["n_queries"] = int(cfg["model"].get("n_queries", 8))
        man["mg_note"] = ("per-query outputs; select gates by "
                          "sigmoid(presence_logit), the flown target by "
                          "argmax(target_logit)")
    return man


def _parity_check(wrapper, inputs, out_path):
    try:
        import numpy as np  # noqa: F401
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed — skipped parity check "
              "(uv pip install onnxruntime)")
        return
    sess = ort.InferenceSession(str(out_path),
                                providers=["CPUExecutionProvider"])
    feeds = {"image": inputs[0].cpu().numpy(), "ego": inputs[1].cpu().numpy(),
             "h_in": inputs[2].cpu().numpy()}
    outs = sess.run(None, feeds)
    with torch.no_grad():
        ref = wrapper(*inputs)
    err = max(float(abs(o - r.detach().cpu().numpy()).max())
              for o, r in zip(outs, ref))
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
