from __future__ import annotations

import numpy as np


class FFTFeatureExtractor:
    """Optional frequency-domain feature extractor hook for future fusion models."""

    def __init__(self, n_bins: int = 32) -> None:
        self.n_bins = n_bins

    def extract(self, image_rgb: np.ndarray) -> np.ndarray:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb must be HxWx3")

        gray = image_rgb.mean(axis=2)
        fft2 = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft2)
        magnitude = np.log1p(np.abs(fft_shift))

        radial_profile = magnitude.mean(axis=0)
        if len(radial_profile) < self.n_bins:
            pad_width = self.n_bins - len(radial_profile)
            radial_profile = np.pad(radial_profile, (0, pad_width), mode="constant")
        else:
            radial_profile = radial_profile[: self.n_bins]
        return radial_profile.astype(np.float32)
