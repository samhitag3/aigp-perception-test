from __future__ import annotations
import torch
from torch import nn
import torchvision.models as tvm


class ResNetBackbone(nn.Module):
    def __init__(self, name="resnet18", pretrained=False, out_dim=256):
        super().__init__()
        weights = None
        if pretrained:
            weights = {
                "resnet18": tvm.ResNet18_Weights.DEFAULT,
                "resnet34": tvm.ResNet34_Weights.DEFAULT,
                "resnet50": tvm.ResNet50_Weights.DEFAULT,
            }[name]
        net = getattr(tvm, name)(weights=weights)
        channels = 512 if name in ("resnet18","resnet34") else 2048
        self.stem = nn.Sequential(net.conv1,net.bn1,net.relu,net.maxpool,net.layer1,net.layer2,net.layer3,net.layer4)
        self.proj = nn.Conv2d(channels,out_dim,1)
        self.mask_up = nn.Sequential(
            nn.ConvTranspose2d(out_dim,out_dim//2,2,2),nn.GELU(),
            nn.ConvTranspose2d(out_dim//2,out_dim,2,2),nn.GELU(),
        )
    def forward(self,x):
        f=self.proj(self.stem(x))
        return f, self.mask_up(f)
