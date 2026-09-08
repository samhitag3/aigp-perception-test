from __future__ import annotations
import math
import torch
from torch import nn
from gatepose.models.blocks import Encoder,ConvGRUCell,PixelDecoder
from gatepose.models.geometry import rotation_6d_to_matrix


def sine_positional_encoding(h,w,d,device):
    y,x=torch.meshgrid(torch.linspace(0,1,h,device=device),torch.linspace(0,1,w,device=device),indexing="ij")
    freqs=torch.arange(max(1,d//4),device=device).float()+1
    pe=torch.cat([torch.sin(2*math.pi*x[...,None]*freqs),torch.cos(2*math.pi*x[...,None]*freqs),torch.sin(2*math.pi*y[...,None]*freqs),torch.cos(2*math.pi*y[...,None]*freqs)],-1)
    if pe.shape[-1]<d: pe=torch.nn.functional.pad(pe,(0,d-pe.shape[-1]))
    return pe[...,:d].reshape(h*w,d)


class GatePoseNetCanonical(nn.Module):
    def __init__(self,cfg:dict):
        super().__init__(); m=cfg["model"]
        d=int(m.get("d_model",128)); q=int(m.get("num_queries",8)); heads=int(m.get("decoder_heads",4)); layers=int(m.get("decoder_layers",2)); track=int(m.get("track_dim",64))
        self.d=d; self.q=q; self.num_kp=8; self.num_vis=int(m.get("num_visibility_classes",5)); self.ego_drop=float(m.get("ego_dropout_probability",0.25))
        self.encoder=Encoder(d); self.ego_mlp=nn.Sequential(nn.Linear(7,d),nn.SiLU(),nn.Linear(d,d)); self.k_mlp=nn.Sequential(nn.Linear(4,d),nn.SiLU(),nn.Linear(d,d))
        self.gru=ConvGRUCell(d); self.queries=nn.Parameter(torch.randn(q,d)*0.02)
        layer=nn.TransformerDecoderLayer(d,heads,dim_feedforward=d*4,batch_first=True,norm_first=True)
        self.decoder=nn.TransformerDecoder(layer,layers)
        self.pixel_decoder=PixelDecoder(d)
        self.mask_embed=nn.Linear(d,d)
        def head(n): return nn.Sequential(nn.Linear(d,d),nn.SiLU(),nn.Linear(d,n))
        self.object_head=head(1); self.box_head=head(4); self.kp_head=head(16); self.kp_conf_head=head(8); self.vis_head=head(8*self.num_vis)
        self.trans_head=head(3); self.rot_head=head(6); self.track_head=head(track); self.target_head=head(1); self.visible_head=head(1); self.depth_head=head(1)

    def step(self,image,ego,intrinsics,h=None):
        feat=self.encoder(image)
        if self.training and self.ego_drop>0:
            keep=(torch.rand((ego.shape[0],1),device=ego.device)>=self.ego_drop).float(); ego=ego*keep
        cond=self.ego_mlp(ego)+self.k_mlp(intrinsics)
        feat=feat+cond[:,:,None,None]
        h=self.gru(feat,h)
        B,C,H,W=h.shape
        mem=h.flatten(2).transpose(1,2)+sine_positional_encoding(H,W,C,h.device)[None]
        query=self.queries[None].expand(B,-1,-1)
        z=self.decoder(query,mem)
        pix=self.pixel_decoder(h)
        mask=torch.einsum("bqd,bdhw->bqhw",self.mask_embed(z),pix)
        box_raw=torch.sigmoid(self.box_head(z)); x1y1=torch.minimum(box_raw[...,:2],box_raw[...,2:]); x2y2=torch.maximum(box_raw[...,:2],box_raw[...,2:]); boxes=torch.cat([x1y1,x2y2],-1)
        rot6=self.rot_head(z)
        trans_raw=self.trans_head(z)
        translation=torch.cat([trans_raw[...,:2], torch.nn.functional.softplus(trans_raw[...,2:3])+0.05], dim=-1)
        return {
            "instances":{
                "mask_logits":mask,
                "object_logits":self.object_head(z).squeeze(-1),
                "boxes":boxes,
                "keypoints":self.kp_head(z).view(B,self.q,8,2),
                "keypoint_logits":self.kp_conf_head(z).view(B,self.q,8),
                "visibility_logits":self.vis_head(z).view(B,self.q,8,self.num_vis),
                "pose":{"translation":translation,"rotation":rot6},
                "track_embeddings":self.track_head(z),
            },
            "auxiliary":{
                "target_logits":self.target_head(z).squeeze(-1),
                "visible_fraction":torch.sigmoid(self.visible_head(z).squeeze(-1)),
                "depth_log":self.depth_head(z).squeeze(-1),
                "rotation_matrix":rotation_6d_to_matrix(rot6),
                "hidden_state":h,
            }
        },h

    def forward(self,images,ego,intrinsics,h=None):
        # images [B,T,3,H,W]
        time=[]
        for t in range(images.shape[1]):
            o,h=self.step(images[:,t],ego[:,t],intrinsics[:,t],h)
            time.append(o)
        inst_keys=["mask_logits","object_logits","boxes","keypoints","keypoint_logits","visibility_logits","track_embeddings"]
        instances={k:torch.stack([o["instances"][k] for o in time],1) for k in inst_keys}
        instances["pose"]={k:torch.stack([o["instances"]["pose"][k] for o in time],1) for k in ["translation","rotation"]}
        aux={k:torch.stack([o["auxiliary"][k] for o in time],1) for k in ["target_logits","visible_fraction","depth_log","rotation_matrix"]}
        aux["hidden_state"]=h
        return {"instances":instances,"auxiliary":aux}
