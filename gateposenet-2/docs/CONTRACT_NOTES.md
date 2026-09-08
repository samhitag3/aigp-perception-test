# Contract notes

## Instance masks

Canonical ground truth and persisted predictions use one single-channel `uint16` PNG per frame:

- `0`: background
- `1..N`: frame-local `mask_id`

`track_id` is separate and persistent across frames.

## Keypoint order

1. `outer_tl`
2. `outer_tr`
3. `outer_br`
4. `outer_bl`
5. `inner_tl`
6. `inner_tr`
7. `inner_br`
8. `inner_bl`

Visibility classes are: `visible`, `occluded`, `out_of_frame`, `behind_camera`, `invalid`.

## Pose

`T_camera_gate` is authoritative. Translation is in meters in camera optical coordinates (+X right, +Y down, +Z forward).

## Training vs persistence

Training outputs logits and continuous tensors. `gatepose.io.decoder.decode_frame()` is the single postprocessing path used for validation/test/video inference and converts those tensors to the persistent canonical prediction contract.
