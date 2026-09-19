from gateperceiver.geometry import fov_from_intrinsics

def test_canonical_camera_fov():
    hfov,vfov=fov_from_intrinsics(640,360,320,320)
    assert abs(hfov-90.0)<1e-6
    assert abs(vfov-58.715507)<1e-5
