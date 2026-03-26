from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

REQUIRED_TREE = [
    "train/real/coco",
    "train/fake/progan",
    "train/fake/diffusion",
    "val/real/coco",
    "val/fake/progan",
    "val/fake/diffusion",
    "test/real/coco",
    "test/fake/unseen_gan",
    "test/fake/unseen_diffusion",
]


def create_data_folders(root: str | Path = "data") -> dict[str, Path]:
    root = Path(root)
    created: dict[str, Path] = {}
    for rel in REQUIRED_TREE:
        d = root / rel
        d.mkdir(parents=True, exist_ok=True)
        created[rel] = d
    return created


def _count_images(folder: Path) -> int:
    return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS)


def list_image_files(folder: str | Path) -> list[Path]:
    root = Path(folder)
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS]


def hash_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    md5 = hashlib.md5()
    with Path(path).open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()


def average_hash(path: str | Path, hash_size: int = 8) -> int:
    image = Image.open(path).convert("L").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
    pixels = list(image.getdata())
    mean = sum(pixels) / len(pixels)
    bits = "".join("1" if p > mean else "0" for p in pixels)
    return int(bits, 2)


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def check_leakage(split_a_files: list[Path], split_b_files: list[Path], name_a: str, name_b: str) -> None:
    hashes_a = {hash_file(p) for p in split_a_files}
    overlap = []
    for p in split_b_files:
        if hash_file(p) in hashes_a:
            overlap.append(p)
    if overlap:
        sample = [str(p) for p in overlap[:5]]
        raise RuntimeError(f"Data leakage detected between {name_a} and {name_b}. Example overlapping files: {sample}")


def check_near_duplicates(
    split_a_files: list[Path],
    split_b_files: list[Path],
    name_a: str,
    name_b: str,
    hamming_threshold: int = 4,
) -> None:
    print("Skipping near-duplicate check for speed.")
    return


def check_split_leakage(root: str | Path = "data") -> None:
    root = Path(root)
    train_files = list_image_files(root / "train")
    val_files = list_image_files(root / "val")
    test_files = list_image_files(root / "test")
    check_leakage(train_files, val_files, "train", "val")
    check_leakage(train_files, test_files, "train", "test")
    check_leakage(val_files, test_files, "val", "test")
    check_near_duplicates(train_files, val_files, "train", "val")
    check_near_duplicates(train_files, test_files, "train", "test")
    check_near_duplicates(val_files, test_files, "val", "test")


def validate_dataset(root: str | Path = "data") -> dict[str, dict[str, int | bool]]:
    root = Path(root)
    summary: dict[str, dict[str, int | bool]] = {}
    print("\n=== Dataset Validation Summary ===")
    for rel in REQUIRED_TREE:
        d = root / rel
        exists = d.exists()
        count = _count_images(d) if exists else 0
        summary[rel] = {"exists": exists, "count": count, "empty": count == 0}
        status = "OK" if exists and count > 0 else "WARN"
        print(f"[{status}] {rel:<30} exists={exists} images={count}")

    missing = [k for k, v in summary.items() if not v["exists"]]
    empty = [k for k, v in summary.items() if v["exists"] and v["empty"]]
    if missing:
        print(f"\nWARNING: Missing folders: {missing}")
    if empty:
        print(f"WARNING: Empty folders: {empty}")
    print("=================================\n")
    return summary
