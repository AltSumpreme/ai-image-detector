from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ViTBranch(nn.Module):
    def __init__(self, out_dim: int = 256, pretrained: bool = True) -> None:
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.backbone = vit_b_16(weights=weights)
        in_dim = self.backbone.heads.head.in_features
        self.backbone.heads = nn.Identity()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.proj(feat)

    def freeze(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_layers(self, num_layers: int = 2) -> None:
        layers = self.backbone.encoder.layers
        for layer in layers[-num_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self.backbone.encoder.ln.parameters():
            p.requires_grad = True
        for p in self.proj.parameters():
            p.requires_grad = True
