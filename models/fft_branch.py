from __future__ import annotations

import torch
import torch.nn as nn


class FFTBranch(nn.Module):
    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fft = torch.fft.fft2(x, norm="ortho")
        mag = torch.abs(torch.fft.fftshift(fft, dim=(-2, -1)))
        log_mag = torch.log1p(mag)
        norm = (log_mag - log_mag.mean(dim=(-2, -1), keepdim=True)) / (log_mag.std(dim=(-2, -1), keepdim=True) + 1e-6)
        feat = self.encoder(norm).flatten(1)
        return self.proj(feat)
