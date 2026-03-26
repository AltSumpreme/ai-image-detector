from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from datasets.data_utils import check_split_leakage, create_data_folders, validate_dataset
from engine.evaluator import Evaluator
from models.main_model import MultiSignalDeepfakeDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multi-signal deepfake detector")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_multisignal.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    create_data_folders(cfg["data_root"])
    validate_dataset(cfg["data_root"])
    check_split_leakage(cfg["data_root"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiSignalDeepfakeDetector().to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])

    evaluator = Evaluator(model, device, image_size=cfg["image_size"])
    seen = evaluator.evaluate("seen", cfg["data_root"], cfg["batch_size"], cfg["num_workers"])
    unseen = evaluator.evaluate("unseen", cfg["data_root"], cfg["batch_size"], cfg["num_workers"])
    degraded = evaluator.evaluate("degraded", cfg["data_root"], cfg["batch_size"], cfg["num_workers"])

    print("Seen:", seen)
    print("Unseen:", unseen)
    print("Degraded:", degraded)

    test_fake = Path(cfg["data_root"]) / "test" / "fake"
    demo_image = next((p for p in test_fake.rglob("*") if p.is_file()), None)
    if demo_image:
        robust = evaluator.robust_predict(demo_image, tta_runs=cfg["tta_runs"])
        print(f"Robust predict example ({demo_image.name}):", robust)


if __name__ == "__main__":
    main()
