from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class SRMBranch(nn.Module):
    def __init__(self, out_dim: int = 256, pretrained: bool = True) -> None:
        super().__init__()
        self.register_buffer("srm_kernels", self._build_srm_kernels())
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Linear(512, out_dim)

    def _build_srm_kernels(self) -> torch.Tensor:
        k1 = torch.tensor([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]], dtype=torch.float32) / 4.0
        k2 = torch.tensor([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=torch.float32) / 12.0
        k3 = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.float32) / 2.0
        kernels = torch.stack([k1, k2, k3], dim=0).unsqueeze(1)
        return kernels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        weight = self.srm_kernels.repeat(c, 1, 1, 1)
        filtered = F.conv2d(x, weight, padding=2, groups=c)
        filtered = filtered.view(b, c, 3, filtered.shape[-2], filtered.shape[-1]).mean(dim=2)
        feat = self.backbone(filtered)
        return self.proj(feat)
