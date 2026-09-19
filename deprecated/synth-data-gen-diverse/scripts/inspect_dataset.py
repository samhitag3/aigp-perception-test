#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description="Summarize camera calibration, trajectory diversity, and speeds.")
p.add_argument("dataset")
a = p.parse_args()
root = Path(a.dataset)

seqs = sorted((root / "sequences").glob("seq_*"))
Ks, resolutions = set(), set()
profiles, styles, tiers = Counter(), Counter(), Counter()
means, mins, maxs, durations, lengths, frames = [], [], [], [], [], []
turns = []

for sd in seqs:
    d = json.loads((sd / "sequence.json").read_text())
    c = d["camera"]
    Ks.add(tuple(tuple(float(x) for x in row) for row in c["intrinsics"]["K"]))
    resolutions.add((int(c["width_px"]), int(c["height_px"])))
    m = d.get("motion", {})
    profiles[m.get("motion_profile", {}).get("name", "unknown")] += 1
    styles[m.get("course_style", "unknown")] += 1
    tiers[d.get("domain_randomization", {}).get("noise_tier", "unknown")] += 1
    for arr, key in [(means, "speed_mean_mps"), (mins, "speed_min_mps"), (maxs, "speed_max_mps"), (durations, "duration_s"), (lengths, "path_length_m")]:
        if key in m:
            arr.append(float(m[key]))
    frames.append(int(d.get("num_frames", 0)))
    for g in d.get("course", {}).get("gate_tracks", []):
        turns.append(abs(float(g.get("turn_from_previous_deg", 0.0))))

def stats(x):
    if not x:
        return None
    x = np.asarray(x, float)
    return {"min": float(x.min()), "mean": float(x.mean()), "median": float(np.median(x)), "p95": float(np.percentile(x, 95)), "max": float(x.max())}

print(json.dumps({
    "sequences": len(seqs),
    "unique_camera_K": len(Ks),
    "camera_K": [list(map(list, K)) for K in sorted(Ks)],
    "unique_resolutions": len(resolutions),
    "resolutions": sorted(resolutions),
    "motion_profiles": profiles,
    "course_styles": styles,
    "noise_tiers": tiers,
    "sequence_frames": stats(frames),
    "speed_mean_mps": stats(means),
    "speed_min_mps": stats(mins),
    "speed_max_mps": stats(maxs),
    "duration_s": stats(durations),
    "path_length_m": stats(lengths),
    "absolute_gate_turn_deg": stats(turns),
}, indent=2, default=dict))
