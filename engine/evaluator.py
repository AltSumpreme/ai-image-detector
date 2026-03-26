from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from datasets.deepfake_dataset import DeepfakeDataset
from models.main_model import MultiSignalDeepfakeDetector
from utils.augmentations import (
    RobustTransforms,
    build_adversarial_like_transform,
    build_degraded_transform,
    build_real_shift_transform,
    build_stress_transform,
)


class Evaluator:
    def __init__(self, model: MultiSignalDeepfakeDetector, device: torch.device, image_size: int = 224) -> None:
        self.model = model
        self.device = device
        self.image_size = image_size

    @staticmethod
    def _metrics(y_true: list[int], y_pred: list[int], src_true: list[str], src_pred: list[int]) -> dict:
        yt = np.array(y_true)
        yp = np.array(y_pred)
        acc = float((yt == yp).mean()) if len(yt) else 0.0
        fake_mask = yt == 1
        real_mask = yt == 0
        fake_rate = float((yp[fake_mask] == 1).mean()) if fake_mask.any() else 0.0
        real_rate = float((yp[real_mask] == 0).mean()) if real_mask.any() else 0.0

        per_src: dict[str, list[int]] = defaultdict(list)
        per_src_cm: dict[str, dict[str, int]] = defaultdict(lambda: {"tn": 0, "fp": 0, "fn": 0, "tp": 0})
        for src, t, p in zip(src_true, y_true, src_pred):
            per_src[src].append(int(t == p))
            if t == 1 and p == 1:
                per_src_cm[src]["tp"] += 1
            elif t == 1 and p == 0:
                per_src_cm[src]["fn"] += 1
            elif t == 0 and p == 1:
                per_src_cm[src]["fp"] += 1
            else:
                per_src_cm[src]["tn"] += 1
        per_src_acc = {k: float(np.mean(v)) for k, v in per_src.items()}

        return {
            "accuracy": acc,
            "fake_detection_rate": fake_rate,
            "real_detection_rate": real_rate,
            "per_source_accuracy": per_src_acc,
            "per_source_confusion": dict(per_src_cm),
        }

    def _predict_loader(self, loader: DataLoader, ablate_branch: str | None = None) -> dict:
        self.model.eval()
        y_true, y_pred, src_true, src_pred = [], [], [], []
        with torch.no_grad():
            for images, labels, source_types in loader:
                out = self.model(images.to(self.device), ablate_branch=ablate_branch)
                preds = (out["probability"] >= 0.5).long().cpu().tolist()
                lbls = labels.tolist()
                y_true.extend(lbls)
                y_pred.extend(preds)
                src_true.extend(source_types)
                src_pred.extend(preds)
        return self._metrics(y_true, y_pred, src_true, src_pred)

    @staticmethod
    def validate_unseen_data(data_root: str | Path) -> dict[str, int]:
        root = Path(data_root)
        required = {
            "unseen_gan": root / "test" / "fake" / "unseen_gan",
            "unseen_diffusion": root / "test" / "fake" / "unseen_diffusion",
        }
        counts: dict[str, int] = {}
        for name, folder in required.items():
            files = [p for p in folder.rglob("*") if p.is_file()]
            counts[name] = len(files)
            print(f"[unseen-count] {name}: {counts[name]}")
            if counts[name] == 0:
                raise RuntimeError(f"Unseen evaluation folder is empty: {folder}")
        return counts

    def evaluate(
        self,
        mode: str,
        data_root: str | Path,
        batch_size: int,
        num_workers: int,
        ablate_branch: str | None = None,
    ) -> dict:
        data_root = Path(data_root)
        if mode == "seen":
            ds = DeepfakeDataset(data_root / "val", transform=RobustTransforms(self.image_size, training=False))
        elif mode == "unseen":
            self.validate_unseen_data(data_root)
            ds = DeepfakeDataset(data_root / "test", transform=RobustTransforms(self.image_size, training=False))
        elif mode == "degraded":
            ds = DeepfakeDataset(data_root / "test", transform=build_degraded_transform(self.image_size))
        elif mode == "real_shift":
            ds = DeepfakeDataset(data_root / "test", transform=build_real_shift_transform(self.image_size))
        elif mode == "stress":
            ds = DeepfakeDataset(data_root / "test", transform=build_stress_transform(self.image_size))
        elif mode == "adversarial":
            ds = DeepfakeDataset(data_root / "test", transform=build_adversarial_like_transform(self.image_size))
        else:
            raise ValueError("mode must be one of: seen, unseen, degraded, real_shift, stress, adversarial")

        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return self._predict_loader(loader, ablate_branch=ablate_branch)

    def evaluate_branch_ablation(self, data_root: str | Path, batch_size: int, num_workers: int, mode: str = "unseen") -> dict[str, dict]:
        results = {}
        for branch in ["srm", "fft", "dwt", "vit", "physical"]:
            results[branch] = self.evaluate(mode, data_root, batch_size, num_workers, ablate_branch=branch)
        return results

    def robust_predict(self, image_path: str | Path, tta_runs: int = 5) -> dict[str, float]:
        image = Image.open(image_path).convert("RGB")
        tta = RobustTransforms(self.image_size, training=True)

        probs = []
        self.model.eval()
        with torch.no_grad():
            for _ in range(tta_runs):
                x = tta(image).unsqueeze(0).to(self.device)
                out = self.model(x)
                probs.append(out["probability"].item())

        avg_prob = float(np.mean(probs))
        return {"probability": avg_prob, "prediction": float(avg_prob >= 0.5)}
