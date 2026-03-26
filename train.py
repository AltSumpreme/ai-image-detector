from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from datasets.data_utils import create_data_folders, validate_dataset
from datasets.deepfake_dataset import DeepfakeDataset
from engine.trainer import Trainer
from utils.augmentations import RobustTransforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-signal deepfake detector")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    set_seed(cfg["seed"])
    create_data_folders(cfg["data_root"])
    validate_dataset(cfg["data_root"])

    train_ds = DeepfakeDataset(Path(cfg["data_root"]) / "train", transform=RobustTransforms(cfg["image_size"], training=True))
    val_ds = DeepfakeDataset(Path(cfg["data_root"]) / "val", transform=RobustTransforms(cfg["image_size"], training=False))

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(cfg, device)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
