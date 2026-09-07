from __future__ import annotations
import torch
from torch import nn


class ConvGRUCell(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, kernel_size: int = 3):
        super().__init__()
        p = kernel_size // 2
        self.hidden_ch = hidden_ch
        self.gates = nn.Conv2d(in_ch + hidden_ch, 2 * hidden_ch, kernel_size, padding=p)
        self.candidate = nn.Conv2d(in_ch + hidden_ch, hidden_ch, kernel_size, padding=p)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None) -> torch.Tensor:
        if h is None:
            h = torch.zeros(x.shape[0], self.hidden_ch, x.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype)
        z, r = torch.sigmoid(self.gates(torch.cat([x, h], dim=1))).chunk(2, dim=1)
        n = torch.tanh(self.candidate(torch.cat([x, r * h], dim=1)))
        return (1 - z) * n + z * h


class ConvGRU(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int):
        super().__init__()
        self.cell = ConvGRUCell(in_ch, hidden_ch)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # x: [B,T,C,H,W]
        h = None
        states: list[torch.Tensor] = []
        for t in range(x.shape[1]):
            h = self.cell(x[:, t], h)
            states.append(h)
        assert h is not None
        return h, states
