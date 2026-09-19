# GateNet venue v2 (AI-GP venue + onboard Arducam)

Drop-in replacement for `gate_perception/model_artifacts/gatenet.onnx` in AIGP_main_flightstack (timothy/perception):
same graph I/O, `image` [batch,3,360,640] float RGB in [0,1] -> `mask` [batch,1,360,640] sigmoid probability.

| held-out set (never trained on) | deployed gatenet.onnx | v1 (venue only) | **v2 (this model)** |
|---|---|---|---|
| real manual flight, 484 frames | 0.702 | 0.630 | **0.790** |
| venue clips, 620 frames | 0.820 | 0.875 | **0.876** |

IoU at 640x360, threshold 0.5; v2 gains mostly recall (0.75 -> 0.82) - it stops missing distant gates.

Trained from Rohan's `runs/gatenet_aigp_6/best.pt` on 6213 frames: the hand-reviewed venue set (pilot lap + DVR) and
the Arducam bags, on the ORIGINAL dark frames. Labels: SAM3 + cleanup; the gate opening is NOT part of the mask and a
double gate is two gates. The labelled dataset itself is in the pCloud folder "2026-09-17_richik_aigp_gate_seg_v1".

Caveat: Arducam labels are automatic (SAM3), so the flight number is agreement with those labels.
