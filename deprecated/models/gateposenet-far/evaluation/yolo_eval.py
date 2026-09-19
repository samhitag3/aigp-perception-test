"""Evaluate a YOLO26 / YOLOE segmentation model.

Reports two things:
  * Ultralytics native metrics (box & mask mAP) via ``model.val`` on the data.yaml
    test split — the standard detection/segmentation benchmark.
  * A common binary-mask IoU/Dice/PR/F1 over data/test (the union of predicted
    instance masks vs the GT mask), so YOLO is directly comparable to GateNet.
"""

from __future__ import annotations

import cv2
import numpy as np

from .common import load_test_pairs, read_gt_mask
from .metrics import MetricAccumulator, mask_metrics


def _load_model(weights, family):
    try:
        import ultralytics  # noqa: F401
    except ImportError as e:
        raise SystemExit("ultralytics not installed. Run: uv sync --extra yolo") from e
    from ultralytics import YOLO, YOLOE
    return (YOLOE if family == "yoloe" else YOLO)(weights)


def _union_mask(result, shape):
    """Combine a YOLO result's instance masks into one binary mask at `shape`."""
    H, W = shape
    out = np.zeros((H, W), np.uint8)
    if getattr(result, "masks", None) is None or result.masks is None:
        return out
    data = result.masks.data.cpu().numpy()  # (n, h, w)
    for m in data:
        mm = cv2.resize(m.astype(np.float32), (W, H))
        out |= (mm > 0.5).astype(np.uint8)
    return out


def evaluate(weights, data_yaml="data/yolo/data.yaml", test_dir="data/test",
             family="yolo26", imgsz=640, device=None, conf=0.25,
             native=True) -> dict:
    model = _load_model(weights, family)
    res = {"backend": family, "weights": str(weights)}

    # 1) native Ultralytics metrics on the test split
    if native:
        try:
            m = model.val(data=data_yaml, split="test", imgsz=imgsz,
                          device=device if device is not None else 0, verbose=False)
            res["box_map50"] = float(getattr(m.box, "map50", float("nan")))
            res["box_map"] = float(getattr(m.box, "map", float("nan")))
            if getattr(m, "seg", None) is not None:
                res["mask_map50"] = float(getattr(m.seg, "map50", float("nan")))
                res["mask_map"] = float(getattr(m.seg, "map", float("nan")))
        except Exception as e:  # pragma: no cover
            res["native_error"] = str(e)

    # 2) common binary-mask metrics vs GateNet test masks
    pairs = load_test_pairs(test_dir)
    acc = MetricAccumulator()
    for _, ip, mp in pairs:
        gt = read_gt_mask(mp)
        r = model.predict(str(ip), imgsz=imgsz, conf=conf,
                          device=device if device is not None else 0, verbose=False)[0]
        pred = _union_mask(r, gt.shape)
        acc.update(mask_metrics(pred, gt))
    res.update(acc.result())
    res["n_images"] = len(pairs)
    return res
