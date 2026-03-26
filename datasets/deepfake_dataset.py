from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset

from datasets.data_utils import VALID_EXTS


def safe_open(path: Path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        print(f"Skipping corrupt image: {path}")
        return None


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label: int
    source_type: str


class DeepfakeDataset(Dataset):
    """Source-aware dataset that recursively scans split folders."""

    def __init__(self, root: str | Path, transform: Callable | None = None, max_samples: int | None = None) -> None:
        self.root = Path(root)
        self.transform = transform
        self.samples = self._scan()
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise RuntimeError(f"No valid images found in {self.root}")

    def _scan(self) -> list[Sample]:
        samples: list[Sample] = []
        for p in sorted(self.root.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in VALID_EXTS:
                continue
            parts = [x.lower() for x in p.parts]
            label = 0 if "real" in parts else 1 if "fake" in parts else -1
            if label == -1:
                continue
            source_type = p.parent.name.lower()
            samples.append(Sample(image_path=p, label=label, source_type=source_type))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = safe_open(sample.image_path)
        if image is None:
            if len(self.samples) == 1:
                raise RuntimeError(f"All images are corrupt under {self.root}")
            return self.__getitem__((idx + 1) % len(self))
        if self.transform:
            image = self.transform(image)
        return image, sample.label, sample.source_type
