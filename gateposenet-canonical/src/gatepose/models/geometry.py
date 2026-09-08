from __future__ import annotations
import torch
import torch.nn.functional as F


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1=d6[...,0:3]; a2=d6[...,3:6]
    b1=F.normalize(a1,dim=-1)
    b2=F.normalize(a2-(b1*a2).sum(-1,keepdim=True)*b1,dim=-1)
    b3=torch.cross(b1,b2,dim=-1)
    return torch.stack((b1,b2,b3),dim=-1)
