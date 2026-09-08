from __future__ import annotations
import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, c1,c2,stride=2):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(c1,c2,3,stride,1,bias=False), nn.BatchNorm2d(c2), nn.SiLU(),
            nn.Conv2d(c2,c2,3,1,1,bias=False), nn.BatchNorm2d(c2), nn.SiLU(),
        )
    def forward(self,x): return self.net(x)


class Encoder(nn.Module):
    def __init__(self,d=128):
        super().__init__()
        self.net=nn.Sequential(ConvBlock(3,32,2),ConvBlock(32,64,2),ConvBlock(64,96,2),ConvBlock(96,d,2))
    def forward(self,x): return self.net(x)


class ConvGRUCell(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.gates=nn.Conv2d(c*2,c*2,3,1,1)
        self.cand=nn.Conv2d(c*2,c,3,1,1)
    def forward(self,x,h):
        if h is None: h=torch.zeros_like(x)
        z,r=torch.sigmoid(self.gates(torch.cat([x,h],1))).chunk(2,1)
        n=torch.tanh(self.cand(torch.cat([x,r*h],1)))
        return (1-z)*n+z*h


class PixelDecoder(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(nn.ConvTranspose2d(d,d,4,2,1),nn.SiLU(),nn.ConvTranspose2d(d,d,4,2,1),nn.SiLU())
    def forward(self,x): return self.net(x)
