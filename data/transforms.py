from __future__ import annotations

import random
from typing import Callable

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


class RandomJPEGCompression:
    """Apply random JPEG compression artifacts to improve robustness."""

    def __init__(self, quality_range: tuple[int, int] = (35, 95), p: float = 0.5) -> None:
        self.quality_range = quality_range
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image

        array = np.array(image)
        bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        quality = random.randint(*self.quality_range)
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, encoded = cv2.imencode(".jpg", bgr, encode_params)
        if not success:
            return image
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)


def get_train_transforms(image_size: int) -> Callable:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
            RandomJPEGCompression(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_eval_transforms(image_size: int) -> Callable:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
