from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

from PIL import Image
from torch.utils.data import Dataset, Subset

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class SampleRecord:
    path: Path
    label: int
    label_name: str


class ImageAIDetectionDataset(Dataset):
    """Dataset with folder layout:

    dataset/
      real/
      ai/
    """

    def __init__(self, root_dir: Path, transform: Callable | None = None) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_to_idx = {"REAL": 0, "AI_GENERATED": 1}
        self.idx_to_class = {index: name for name, index in self.class_to_idx.items()}
        self.samples = self._build_index()

    def _build_index(self) -> List[SampleRecord]:
        split_dirs = {
            "REAL": self.root_dir / "real",
            "AI_GENERATED": self.root_dir / "ai",
        }
        samples: List[SampleRecord] = []
        for class_name, class_dir in split_dirs.items():
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")

            for path in sorted(class_dir.rglob("*")):
                if not path.is_file():
                    continue
                if path.suffix.lower() in VALID_EXTENSIONS:
                    samples.append(
                        SampleRecord(
                            path=path,
                            label=self.class_to_idx[class_name],
                            label_name=class_name,
                        )
                    )

        if not samples:
            raise ValueError(f"No valid image files found under {self.root_dir}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, sample.label

    def get_labels(self) -> List[int]:
        return [sample.label for sample in self.samples]

    def class_counts(self) -> Dict[str, int]:
        counts = {"REAL": 0, "AI_GENERATED": 0}
        for sample in self.samples:
            counts[sample.label_name] += 1
        return counts


def build_train_val_indices(
    dataset: Sequence,
    val_split: float,
    seed: int,
) -> tuple[Subset, Subset]:
    """Return deterministic, stratified train/validation subsets.

    Stratification keeps class balance stable for binary REAL/AI labels.
    """
    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split must be in (0, 1)")

    if not hasattr(dataset, "samples"):
        raise TypeError("dataset must expose a `samples` attribute for stratified splitting")

    label_to_indices: Dict[int, List[int]] = {}
    for index, sample in enumerate(dataset.samples):
        label_to_indices.setdefault(sample.label, []).append(index)

    rng = random.Random(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []

    for _, indices in label_to_indices.items():
        shuffled = indices.copy()
        rng.shuffle(shuffled)

        val_count = max(1, int(len(shuffled) * val_split))
        val_indices.extend(shuffled[:val_count])
        train_indices.extend(shuffled[val_count:])

    if not train_indices or not val_indices:
        raise ValueError("Split produced an empty subset; provide more data or adjust val_split")

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return Subset(dataset, train_indices), Subset(dataset, val_indices)
