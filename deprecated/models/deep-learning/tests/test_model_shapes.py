import torch
from gateperceiver.models import GatePerceiver

def test_forward_shapes():
    cfg={"model":{"backbone":"resnet18","pretrained_backbone":False,"d_model":64,"nheads":4,"temporal_layers":1,"decoder_layers":1,"num_queries":5,"track_dim":16,"keypoint_margin":0.5}}
    m=GatePerceiver(cfg).eval(); rgb=torch.rand(1,2,3,96,160); K=torch.tensor([[[[80.,0,80.],[0,80.,48.],[0,0,1.]],[[80.,0,80.],[0,80.,48.],[0,0,1.]]]])
    with torch.no_grad(): out=m(rgb,K)
    assert out["pred_logits"].shape==(1,2,5)
    assert out["pred_masks"].shape==(1,2,5,96,160)
    assert out["pred_keypoints"].shape==(1,2,5,8,2)
    assert out["pred_visibility"].shape==(1,2,5,8,4)
    assert out["pred_translation"].shape==(1,2,5,3)
    assert out["pred_rotation6d"].shape==(1,2,5,6)
