from gateperception.utils.config import load_yaml
from gateperception.geometry.camera import fov_from_intrinsics
c=load_yaml('configs/camera.yaml')['camera']
h,v=fov_from_intrinsics(c['width_px'],c['height_px'],c['fx'],c['fy'])
print(f"K = [[{c['fx']},0,{c['cx']}],[0,{c['fy']},{c['cy']}],[0,0,1]]")
print(f"HFOV = {h:.6f} deg")
print(f"VFOV = {v:.6f} deg")
