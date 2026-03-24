from __future__ import annotations

from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    efficientnet_b0,
    efficientnet_b1,
)


def build_efficientnet(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
    elif model_name == "efficientnet_b1":
        weights = EfficientNet_B1_Weights.DEFAULT if pretrained else None
        model = efficientnet_b1(weights=weights)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
