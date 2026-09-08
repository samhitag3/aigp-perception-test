from __future__ import annotations
import argparse,torch
from gatepose.utils.config import load_config
from gatepose.models import build_model
p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--output',required=True); a=p.parse_args(); cfg=load_config(a.config); model=build_model(cfg).cpu().eval(); ck=torch.load(a.checkpoint,map_location='cpu'); model.load_state_dict(ck['model'] if 'model' in ck else ck)
class Step(torch.nn.Module):
    def __init__(self,m): super().__init__(); self.m=m
    def forward(self,image,ego,intrinsics,h):
        o,h2=self.m.step(image,ego,intrinsics,h); i=o['instances']; a=o['auxiliary']; return i['mask_logits'],i['object_logits'],i['boxes'],i['keypoints'],i['keypoint_logits'],i['visibility_logits'],i['pose']['translation'],i['pose']['rotation'],i['track_embeddings'],a['target_logits'],a['visible_fraction'],a['depth_log'],h2
w=Step(model); image=torch.zeros(1,3,cfg['camera']['model_height'],cfg['camera']['model_width']); ego=torch.zeros(1,7); intr=torch.zeros(1,4); h=torch.zeros(1,cfg['model']['d_model'],cfg['camera']['model_height']//16,cfg['camera']['model_width']//16)
torch.onnx.export(w,(image,ego,intr,h),a.output,input_names=['image','ego','intrinsics','h'],output_names=['mask_logits','object_logits','boxes','keypoints','keypoint_logits','visibility_logits','translation','rotation6d','track_embeddings','target_logits','visible_fraction','depth_log','h_out'],opset_version=18,dynamic_axes={'image':{0:'batch'},'ego':{0:'batch'},'intrinsics':{0:'batch'},'h':{0:'batch'}}); print(a.output)
