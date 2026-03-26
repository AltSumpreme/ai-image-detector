from __future__ import annotations

from pathlib import Path

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
