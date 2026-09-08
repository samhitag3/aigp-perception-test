from gatepose.utils.camera import fov_from_intrinsics,letterbox_params

def test_aigp_camera():
    h,v=fov_from_intrinsics(640,360,320,320)
    assert abs(h-90)<1e-6
    assert 58.7 < v < 58.8
    tr=letterbox_params(640,360,320,192)
    assert tr.scale==0.5 and tr.pad_y==6 and tr.pad_x==0
