from __future__ import annotations

import io
import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class RandomJPEGCompression:
    def __init__(self, quality_range: tuple[int, int] = (30, 100), p: float = 0.5) -> None:
        self.quality_range = quality_range
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class AddGaussianNoise:
    def __init__(self, sigma_range: tuple[float, float] = (0.0, 0.04), p: float = 0.5) -> None:
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return tensor
        sigma = random.uniform(*self.sigma_range)
        noise = torch.randn_like(tensor) * sigma
        return torch.clamp(tensor + noise, 0.0, 1.0)


class EvalDegradation:
    def __init__(self, jpeg_quality: int = 50, blur_kernel: int = 5) -> None:
        self.jpeg_quality = jpeg_quality
        self.blur = transforms.GaussianBlur(kernel_size=blur_kernel)

    def __call__(self, image: Image.Image) -> Image.Image:
        image = self.blur(image)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self.jpeg_quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


def build_train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
            RandomJPEGCompression((30, 100), p=0.5),
            transforms.ToTensor(),
            AddGaussianNoise((0.0, 0.03), p=0.5),
        ]
    )


def build_eval_transform(image_size: int = 224, degraded: bool = False) -> transforms.Compose:
    ops: list = [transforms.Resize((image_size, image_size))]
    if degraded:
        ops.append(EvalDegradation(jpeg_quality=50, blur_kernel=5))
    ops.extend([transforms.ToTensor()])
    return transforms.Compose(ops)


def apply_stochastic_tta(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    tta = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
            RandomJPEGCompression((35, 95), p=0.5),
            transforms.ToTensor(),
            AddGaussianNoise((0.0, 0.02), p=0.4),
        ]
    )
    return tta(image)
