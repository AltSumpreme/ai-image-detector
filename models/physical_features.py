from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicalFeatures(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)

        mag = torch.sqrt(gx.square() + gy.square() + 1e-6)
        ang = torch.atan2(gy, gx + 1e-6)

        edge_consistency = mag.mean(dim=(2, 3))
        edge_spread = mag.std(dim=(2, 3))
        grad_dir_var = ang.var(dim=(2, 3))

        light_global = gray.mean(dim=(2, 3))
        left = gray[:, :, :, : gray.shape[-1] // 2].mean(dim=(2, 3))
        right = gray[:, :, :, gray.shape[-1] // 2 :].mean(dim=(2, 3))
        light_asym = torch.abs(left - right)

        return torch.cat([edge_consistency, edge_spread, grad_dir_var, light_global, light_asym], dim=1)
