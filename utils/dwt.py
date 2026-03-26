from __future__ import annotations

import torch
import torch.nn.functional as F


def haar_dwt_2d(x: torch.Tensor) -> torch.Tensor:
    """Compute single-level Haar DWT per channel.

    Args:
        x: Tensor of shape [B, C, H, W].

    Returns:
        Tensor of shape [B, 4*C, H/2, W/2] in LL, LH, HL, HH order.
    """
    if x.dim() != 4:
        raise ValueError("Expected [B, C, H, W] tensor")

    ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device=x.device, dtype=x.dtype)
    lh = torch.tensor([[0.5, 0.5], [-0.5, -0.5]], device=x.device, dtype=x.dtype)
    hl = torch.tensor([[0.5, -0.5], [0.5, -0.5]], device=x.device, dtype=x.dtype)
    hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], device=x.device, dtype=x.dtype)

    kernels = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # [4,1,2,2]
    b, c, _, _ = x.shape
    weight = kernels.repeat(c, 1, 1, 1)  # [4*C,1,2,2]
    return F.conv2d(x, weight, stride=2, groups=c)
