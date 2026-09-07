# Contract notes

## Canonical source mask

One `uint16` PNG is stored per frame. Nonzero values are separate **frame-local instances**, not a union mask.

```text
0 background
1 gate instance 1
2 gate instance 2
...
```

A model needing a binary mask for one gate uses `instance_mask == mask_id`. A union model uses `instance_mask > 0`.

## Persistent identity

`mask_id` is frame-local. `track_id` is temporal and must be used for identity across frames.

## Eight keypoints

Fixed physical order:

```text
outer_tl, outer_tr, outer_br, outer_bl,
inner_tl, inner_tr, inner_br, inner_bl
```

Occluded and out-of-frame points retain mathematically valid projected coordinates whenever possible. Their visibility state changes; their physical identity does not.

## Model raw output

Segmentation training uses continuous per-query logits, not an integer PNG:

```text
object_logits      [B,Q]
boxes              [B,Q,4]
mask_logits         [B,Q,H/4,W/4]
track_embeddings    [B,Q,D]
```

The common decoder creates the integer PNG only for validation/deployment.

Keypoint raw output:

```text
keypoints           [B,8,2]
visibility_logits   [B,8,4]
```

## Memory-only prediction

A short segmentation dropout can yield a temporal gate record with `mask_id=null`. This means the system predicts the track/keypoints/pose from recent memory but makes **no claim that gate pixels are currently visible**. Such a track is absent from the current instance PNG.
