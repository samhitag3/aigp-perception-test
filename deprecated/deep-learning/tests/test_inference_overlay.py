import numpy as np
from gateperceiver.inference.overlay import resize_decoded_prediction, draw_prediction_overlay


def test_resize_prediction_and_overlay():
    mask = np.zeros((10, 20), np.uint16)
    mask[2:6, 4:9] = 1
    instances = [{
        "mask_id": 1,
        "track_id": "track_0001",
        "detection_score": 0.9,
        "box_xyxy_px": [4.0, 2.0, 9.0, 6.0],
        "visible_area_px": 20,
        "keypoints": {"outer_tl": {"xy_px": [4.0, 2.0], "confidence": None, "visibility": "visible"}},
        "pose": {"depth_camera_z_m": 3.0},
    }]
    resized, scaled = resize_decoded_prediction(mask, instances, 40, 20)
    assert resized.shape == (20, 40)
    assert scaled[0]["box_xyxy_px"] == [8.0, 4.0, 18.0, 12.0]
    assert scaled[0]["keypoints"]["outer_tl"]["xy_px"] == [8.0, 4.0]
    frame = np.zeros((20, 40, 3), np.uint8)
    vis = draw_prediction_overlay(frame, resized, scaled, {"inference_ms": 5.0, "postprocess_ms": 1.0, "total_ms": 6.0}, 0, 30.0)
    assert vis.shape == frame.shape
    assert np.any(vis != 0)
