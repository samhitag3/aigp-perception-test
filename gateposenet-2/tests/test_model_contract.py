import torch,yaml
from gatepose.models import build_model

def test_shapes():
    cfg=yaml.safe_load(open('configs/base.yaml'))
    m=build_model(cfg); x=torch.zeros(2,3,3,192,320); e=torch.zeros(2,3,7); k=torch.zeros(2,3,4); o=m(x,e,k)
    assert o['instances']['mask_logits'].shape[:3]==(2,3,8)
    assert o['instances']['keypoints'].shape==(2,3,8,8,2)
    assert o['instances']['visibility_logits'].shape==(2,3,8,8,5)
    assert o['instances']['pose']['translation'].shape==(2,3,8,3)
