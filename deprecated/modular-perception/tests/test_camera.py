from gateperception.geometry.camera import fov_from_intrinsics


def test_camera_fov():
    h, v = fov_from_intrinsics(640, 360, 320.0, 320.0)
    assert abs(h - 90.0) < 1e-6
    assert abs(v - 58.7155) < 1e-3
