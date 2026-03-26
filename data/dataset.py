from __future__ import annotations

import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests
from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int
    source: str


class MultiSourceImageDataset(Dataset):
    """Loads data from:

    data/train/real
    data/train/fake
    """

    def __init__(self, real_dir: str | Path, fake_dir: str | Path, transform: Callable | None = None) -> None:
        self.real_dir = Path(real_dir)
        self.fake_dir = Path(fake_dir)
        self.transform = transform
        self.samples = self._index()

    def _index(self) -> list[ImageRecord]:
        samples: list[ImageRecord] = []
        for root, label, source in [(self.real_dir, 0, "real"), (self.fake_dir, 1, "fake")]:
            if not root.exists():
                continue
            for p in sorted(root.rglob("*")):
                if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS:
                    samples.append(ImageRecord(path=p, label=label, source=source))
        if not samples:
            raise RuntimeError("No images found in dataset directories")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        rec = self.samples[idx]
        img = Image.open(rec.path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, rec.label


def _download_single_image(url: str, output_path: Path, timeout: int = 20) -> bool:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        output_path.write_bytes(r.content)
        return True
    except Exception:
        return False


def download_coco_subset(output_dir: str | Path, subset_size: int = 5000, seed: int = 42) -> None:
    """Download a COCO image subset using COCO URLs from official annotations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    zip_path = output_dir / "annotations_trainval2017.zip"
    if not zip_path.exists():
        resp = requests.get(ann_url, timeout=120)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)

    import zipfile
    import json

    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open("annotations/instances_train2017.json") as fh:
            data = json.load(fh)

    images = data["images"]
    rng = random.Random(seed)
    rng.shuffle(images)
    selected = images[:subset_size]

    def _job(meta: dict) -> bool:
        file_name = meta["file_name"]
        url = meta["coco_url"]
        out_path = output_dir / f"coco_{file_name}"
        if out_path.exists():
            return True
        return _download_single_image(url, out_path)

    with ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(_job, selected))

    success = sum(1 for x in results if x)
    print(f"COCO downloaded: {success}/{len(selected)} -> {output_dir}")


def download_diffusiondb_subset(output_dir: str | Path, config_name: str = "large_random_1k", max_images: int = 1000) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("poloclub/diffusiondb", config_name, split="train")
    for i, row in enumerate(ds):
        if i >= max_images:
            break
        img = row["image"]
        out_path = output_dir / f"diffusiondb_{i:06d}.png"
        if not out_path.exists():
            img.save(out_path)

    print(f"DiffusionDB subset saved to {output_dir}")


def copy_gan_samples(source_dir: str | Path, output_dir: str | Path, max_images: int | None = None) -> None:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in sorted(source_dir.rglob("*")) if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
    if max_images is not None:
        files = files[:max_images]
    for idx, f in enumerate(files):
        dst = output_dir / f"gan_{idx:06d}{f.suffix.lower()}"
        if not dst.exists():
            shutil.copy2(f, dst)
    print(f"Copied {len(files)} GAN images to {output_dir}")
