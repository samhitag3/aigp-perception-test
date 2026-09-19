from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .backbone import ResNetBackbone


class MLP(nn.Module):
    def __init__(self, dims, final_activation=None):
        super().__init__(); layers=[]
        for a,b in zip(dims[:-1],dims[1:]):
            layers.append(nn.Linear(a,b))
            if b != dims[-1]: layers.append(nn.GELU())
        self.net=nn.Sequential(*layers); self.final_activation=final_activation
    def forward(self,x):
        x=self.net(x)
        return self.final_activation(x) if self.final_activation else x


class GatePerceiver(nn.Module):
    """Temporal multi-task DETR-like neural network for gate perception."""
    def __init__(self, cfg: dict):
        super().__init__(); m=cfg["model"]
        d=int(m.get("d_model",256)); heads=int(m.get("nheads",8)); q=int(m.get("num_queries",10)); layers=int(m.get("decoder_layers",3)); t_layers=int(m.get("temporal_layers",2))
        self.num_queries=q; self.d_model=d; self.keypoint_margin=float(m.get("keypoint_margin",0.5))
        self.backbone=ResNetBackbone(m.get("backbone","resnet18"),bool(m.get("pretrained_backbone",False)),d)
        enc_layer=nn.TransformerEncoderLayer(d,heads,dim_feedforward=4*d,batch_first=True,norm_first=True)
        self.temporal=nn.TransformerEncoder(enc_layer,t_layers)
        dec_layer=nn.TransformerDecoderLayer(d,heads,dim_feedforward=4*d,batch_first=True,norm_first=True)
        self.decoder=nn.TransformerDecoder(dec_layer,layers)
        self.query_embed=nn.Embedding(q,d)
        self.camera_mlp=MLP([9,d,d])
        self.object_head=nn.Linear(d,1)
        self.box_head=MLP([d,d,4])
        self.kp_head=MLP([d,d,16])
        self.vis_head=MLP([d,d,8*4])
        self.translation_head=MLP([d,d,3])
        self.rotation_head=MLP([d,d,6])
        self.track_head=MLP([d,d,int(m.get("track_dim",64))])
        self.mask_query=nn.Linear(d,d)

    def forward(self, rgb: torch.Tensor, intrinsics: torch.Tensor, gate_geometry: torch.Tensor | None = None):
        B,T,C,H,W=rgb.shape; x=rgb.reshape(B*T,C,H,W)
        feat,mask_feat=self.backbone(x); _,D,h,w=feat.shape
        pooled=feat.mean((-2,-1)).reshape(B,T,D)
        K=intrinsics.reshape(B*T,3,3)
        cam=torch.stack([K[:,0,0]/W,K[:,1,1]/H,K[:,0,2]/W,K[:,1,2]/H],-1).reshape(B,T,4)
        if gate_geometry is None:
            geom=rgb.new_tensor([2.7,2.7,1.5,1.5,0.26]).view(1,1,5).expand(B,T,-1)
        else:
            geom=gate_geometry[:,None,:].expand(B,T,-1)
        frame_ctx=self.temporal(pooled+self.camera_mlp(torch.cat([cam,geom],-1)))
        memory=feat.flatten(2).transpose(1,2).reshape(B,T,h*w,D)
        memory=memory+frame_ctx[:,:,None,:]
        q=self.query_embed.weight.view(1,1,self.num_queries,D).expand(B,T,-1,-1)+frame_ctx[:,:,None,:]
        dec=self.decoder(q.reshape(B*T,self.num_queries,D),memory.reshape(B*T,h*w,D)).reshape(B,T,self.num_queries,D)
        obj=self.object_head(dec).squeeze(-1)
        boxes=torch.sigmoid(self.box_head(dec))
        raw_kp=self.kp_head(dec).reshape(B,T,self.num_queries,8,2)
        # Allows coordinates outside frame: [-margin, 1+margin]
        kp=0.5 + (0.5+self.keypoint_margin)*torch.tanh(raw_kp)
        vis=self.vis_head(dec).reshape(B,T,self.num_queries,8,4)
        tr=self.translation_head(dec)
        # depth should be positive; x/y remain signed
        tr=torch.cat([tr[...,:2],F.softplus(tr[...,2:])+1e-3],-1)
        rot=self.rotation_head(dec)
        emb=F.normalize(self.track_head(dec),dim=-1)
        mq=self.mask_query(dec).reshape(B*T,self.num_queries,D)
        masks=torch.einsum("bqd,bdhw->bqhw",mq,mask_feat).reshape(B,T,self.num_queries,mask_feat.shape[-2],mask_feat.shape[-1])
        masks=F.interpolate(masks.reshape(B*T*self.num_queries,1,*masks.shape[-2:]),size=(H,W),mode="bilinear",align_corners=False).reshape(B,T,self.num_queries,H,W)
        # Flat keys are used internally; `instances` is the canonical raw training-time adapter contract.
        return {
            "pred_logits":obj,"pred_masks":masks,"pred_boxes":boxes,"pred_keypoints":kp,
            "pred_visibility":vis,"pred_translation":tr,"pred_rotation6d":rot,"track_embeddings":emb,
            "instances": {
                "mask_logits": masks,
                "object_logits": obj,
                "boxes": boxes,
                "keypoints": kp,
                "visibility_logits": vis,
                "pose": {"translation": tr, "rotation6d": rot},
                "track_embeddings": emb,
            },
            "auxiliary": {},
        }
