#!/usr/bin/env python3
"""Evaluate binary GateNet predictions against instance-ID ground-truth masks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def index_images(directory: Path) -> dict[str, Path]:
    files = {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    }
    if not files:
        raise RuntimeError(f"No mask images found in {directory}")
    return files


def safe_div(numerator: float, denominator: float, empty_value: float = 0.0) -> float:
    return numerator / denominator if denominator else empty_value


def metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    # An empty prediction on an empty GT frame is considered a perfect frame.
    both_empty = tp + fp + fn == 0
    iou = safe_div(tp, tp + fp + fn, 1.0 if both_empty else 0.0)
    dice = safe_div(2 * tp, 2 * tp + fp + fn, 1.0 if both_empty else 0.0)
    precision = safe_div(tp, tp + fp, 1.0 if both_empty else 0.0)
    recall = safe_div(tp, tp + fn, 1.0 if both_empty else 0.0)
    return {
        "iou": iou,
        "dice_f1": dice,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": safe_div(fp, fp + tn),
        "false_negative_rate": safe_div(fn, fn + tp),
        "pixel_accuracy": safe_div(tp + tn, tp + fp + fn + tn),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--pred-threshold",
        type=float,
        default=127.0,
        help="Prediction pixel threshold. Default is 127 for 0/255 PNG masks.",
    )
    args = parser.parse_args()

    predictions = index_images(args.pred_dir)
    ground_truth = index_images(args.gt_dir)
    common = sorted(predictions.keys() & ground_truth.keys())
    missing_predictions = sorted(ground_truth.keys() - predictions.keys())
    extra_predictions = sorted(predictions.keys() - ground_truth.keys())

    if not common:
        raise RuntimeError("No prediction and ground-truth filenames share the same stem")
    if missing_predictions:
        preview = ", ".join(missing_predictions[:10])
        raise RuntimeError(
            f"Missing predictions for {len(missing_predictions)} GT masks; first: {preview}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    for index, stem in enumerate(common, 1):
        pred_raw = cv2.imread(str(predictions[stem]), cv2.IMREAD_UNCHANGED)
        gt_raw = cv2.imread(str(ground_truth[stem]), cv2.IMREAD_UNCHANGED)
        if pred_raw is None or gt_raw is None:
            raise RuntimeError(f"Could not read prediction or GT mask for {stem}")

        if pred_raw.ndim == 3:
            pred_raw = np.max(pred_raw, axis=2)
        if gt_raw.ndim == 3:
            gt_raw = np.max(gt_raw, axis=2)
        if pred_raw.shape != gt_raw.shape:
            pred_raw = cv2.resize(
                pred_raw,
                (gt_raw.shape[1], gt_raw.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        pred = pred_raw > args.pred_threshold
        # GT is an instance-ID image: 0=background and every nonzero ID=gate.
        gt = gt_raw != 0

        tp = int(np.count_nonzero(pred & gt))
        fp = int(np.count_nonzero(pred & ~gt))
        fn = int(np.count_nonzero(~pred & gt))
        tn = int(np.count_nonzero(~pred & ~gt))
        counts = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
        for key in totals:
            totals[key] += counts[key]

        rows.append({"frame": stem, **counts, **metrics(tp, fp, fn, tn)})
        if index % 100 == 0 or index == len(common):
            print(f"[{index}/{len(common)}] {stem}")

    metric_names = list(metrics(0, 0, 0, 1).keys())
    micro = metrics(**totals)
    macro = {
        name: float(np.mean([float(row[name]) for row in rows]))
        for name in metric_names
    }
    summary = {
        "num_evaluated": len(common),
        "num_ground_truth": len(ground_truth),
        "num_predictions": len(predictions),
        "extra_prediction_count": len(extra_predictions),
        "prediction_threshold": args.pred_threshold,
        "ground_truth_conversion": "gt != 0 (instance IDs converted to union foreground)",
        "micro_pixel_metrics": micro,
        "macro_frame_metrics": macro,
        "pixel_confusion_counts": totals,
    }

    csv_path = args.out_dir / "per_frame_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = args.out_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\nDataset evaluation")
    print(f"  frames:             {len(common)}")
    print(f"  micro IoU:          {micro['iou']:.6f}")
    print(f"  macro IoU:          {macro['iou']:.6f}")
    print(f"  micro Dice/F1:      {micro['dice_f1']:.6f}")
    print(f"  micro precision:    {micro['precision']:.6f}")
    print(f"  micro recall:       {micro['recall']:.6f}")
    print(f"  false-positive rate:{micro['false_positive_rate']:10.6f}")
    print(f"  false-negative rate:{micro['false_negative_rate']:10.6f}")
    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
