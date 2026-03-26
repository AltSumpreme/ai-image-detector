from __future__ import annotations

import argparse
from pathlib import Path

from data.dataset import copy_gan_samples, download_coco_subset, download_diffusiondb_subset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download and stage real/fake datasets")
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--coco-count", type=int, default=5000)
    p.add_argument("--diffusion-config", type=str, default="large_random_1k")
    p.add_argument("--diffusion-count", type=int, default=1000)
    p.add_argument("--gan-train-dir", type=str, default=None)
    p.add_argument("--gan-test-unseen-dir", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.data_root)

    train_real = root / "train" / "real"
    train_fake = root / "train" / "fake"
    unseen_fake = root / "test_unseen" / "fake"

    download_coco_subset(train_real, subset_size=args.coco_count)
    download_diffusiondb_subset(train_fake, config_name=args.diffusion_config, max_images=args.diffusion_count)

    if args.gan_train_dir:
        copy_gan_samples(args.gan_train_dir, train_fake)
    if args.gan_test_unseen_dir:
        copy_gan_samples(args.gan_test_unseen_dir, unseen_fake)


if __name__ == "__main__":
    main()
