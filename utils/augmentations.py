from __future__ import annotations

import io
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms


def add_noise_tensor(img: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    noise = torch.randn_like(img) * sigma
    return torch.clamp(img + noise, 0.0, 1.0)


class RobustTransforms:
    def __init__(self, image_size: int = 224, training: bool = True) -> None:
        self.image_size = image_size
        self.training = training
        self.pre_base = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5 if training else 0.0),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05) if training else transforms.Lambda(lambda x: x),
            ]
        )
        self.to_tensor = transforms.ToTensor()

    def jpeg_compress(self, img: Image.Image) -> Image.Image:
        quality = random.randint(30, 100)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def gaussian_blur(self, img: Image.Image) -> Image.Image:
        k = random.choice([3, 5, 7])
        return img.filter(ImageFilter.GaussianBlur(radius=k // 2))

    def add_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        sigma = random.uniform(0.0, 0.04)
        return add_noise_tensor(tensor, sigma=sigma)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = self.pre_base(img)
        if self.training:
            ops = ["jpeg", "blur", "noise"]
            k = random.randint(1, 3)
            selected = set(random.sample(ops, k=k))
            if "jpeg" in selected:
                img = self.jpeg_compress(img)
            if "blur" in selected:
                img = self.gaussian_blur(img)
            tensor = self.to_tensor(img)
            if "noise" in selected:
                tensor = self.add_noise(tensor)
            return tensor

        return self.to_tensor(img)


def build_degraded_transform(image_size: int = 224) -> transforms.Compose:
    def _degrade(img: Image.Image) -> Image.Image:
        rt = RobustTransforms(image_size=image_size, training=True)
        img = rt.jpeg_compress(img)
        img = rt.gaussian_blur(img)
        return img

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Lambda(_degrade),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: add_noise_tensor(t, sigma=0.05)),
        ]
    )


def build_real_shift_transform(image_size: int = 224) -> transforms.Compose:
    def _shift(img: Image.Image) -> Image.Image:
        rt = RobustTransforms(image_size=image_size, training=True)
        img = rt.jpeg_compress(img)
        img = rt.jpeg_compress(img)  # repeated social-media-like recompression
        img = img.resize((image_size // 2, image_size // 2), Image.Resampling.BILINEAR).resize((image_size, image_size), Image.Resampling.BILINEAR)
        img = rt.gaussian_blur(img)
        return img

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Lambda(_shift),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: add_noise_tensor(t, sigma=0.03)),
        ]
    )


def build_stress_transform(image_size: int = 224) -> transforms.Compose:
    def _stress(img: Image.Image) -> Image.Image:
        for _ in range(3):
            q = random.randint(10, 30)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        img = img.resize((image_size // 2, image_size // 2), Image.Resampling.BILINEAR).resize((image_size, image_size), Image.Resampling.BILINEAR)
        img = img.filter(ImageFilter.GaussianBlur(radius=3.0))
        return img

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Lambda(_stress),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: add_noise_tensor(t, sigma=0.06)),
        ]
    )


def build_adversarial_like_transform(image_size: int = 224) -> transforms.Compose:
    def _adv(img: Image.Image) -> Image.Image:
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.8)
        arr = torch.from_numpy(np.array(img)).float() / 255.0
        arr = torch.clamp(arr + torch.randn_like(arr) * 0.01, 0.0, 1.0)
        img = Image.fromarray((arr.numpy() * 255).astype("uint8"))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(25, 50))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        return img

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Lambda(_adv),
            transforms.ToTensor(),
        ]
    )
