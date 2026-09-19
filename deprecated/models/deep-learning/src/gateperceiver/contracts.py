from __future__ import annotations

KEYPOINT_NAMES = [
    "outer_tl", "outer_tr", "outer_br", "outer_bl",
    "inner_tl", "inner_tr", "inner_br", "inner_bl",
]

VISIBILITY_TO_INDEX = {
    "invalid": 0,
    "behind_camera": 0,
    "visible": 1,
    "occluded": 2,
    "out_of_frame": 3,
}
INDEX_TO_VISIBILITY = {
    0: "invalid",
    1: "visible",
    2: "occluded",
    3: "out_of_frame",
}

CAMERA_DEFAULT = {
    "width_px": 640,
    "height_px": 360,
    "fx": 320.0,
    "fy": 320.0,
    "cx": 320.0,
    "cy": 180.0,
}

SCHEMA_VERSION = "1.0.0"
