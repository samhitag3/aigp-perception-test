"""Evaluate a GateNet checkpoint on a GateNet-format test set (data/test)."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from gatenet.model import GateNetInference, build_gatenet
from gatenet.utils import load_checkpoint, load_config, pick_device

from .common import load_test_pairs, read_gt_mask
from .metrics import MetricAccumulator, mask_metrics


def evaluate(checkpoint, config, test_dir="data/test", device=None,
             threshold=0.5) -> dict:
    cfg = load_config(config)
    dev = pick_device(device)
    H, W = cfg["data"]["height"], cfg["data"]["width"]

    net = build_gatenet(cfg["model"])
    load_checkpoint(checkpoint, net, map_location=dev)
    model = GateNetInference(net, threshold=None).to(dev).eval()

    pairs = load_test_pairs(test_dir)
    acc = MetricAccumulator()
    with torch.no_grad():
        for _, ip, mp in pairs:
            bgr = cv2.imread(str(ip), cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            inp = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
            t = torch.from_numpy(inp).permute(2, 0, 1).float()[None].to(dev) / 255.0
            prob = model(t)[0, 0].cpu().numpy()

            gt = read_gt_mask(mp)
            prob = cv2.resize(prob, (gt.shape[1], gt.shape[0]))
            pred = (prob > threshold).astype(np.uint8)
            acc.update(mask_metrics(pred, gt))

    res = acc.result()
    res["n_images"] = len(pairs)
    res["backend"] = "gatenet"
    return res
