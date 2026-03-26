from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from datasets.deepfake_dataset import DeepfakeDataset
from models.main_model import MultiSignalDeepfakeDetector
from utils.augmentations import RobustTransforms, build_degraded_transform


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
        for src, t, p in zip(src_true, y_true, src_pred):
            per_src[src].append(int(t == p))
        per_src_acc = {k: float(np.mean(v)) for k, v in per_src.items()}

        return {
            "accuracy": acc,
            "fake_detection_rate": fake_rate,
            "real_detection_rate": real_rate,
            "per_source_accuracy": per_src_acc,
        }

    def _predict_loader(self, loader: DataLoader) -> dict:
        self.model.eval()
        y_true, y_pred, src_true, src_pred = [], [], [], []
        with torch.no_grad():
            for images, labels, source_types in loader:
                out = self.model(images.to(self.device))
                preds = (out["probability"] >= 0.5).long().cpu().tolist()
                lbls = labels.tolist()
                y_true.extend(lbls)
                y_pred.extend(preds)
                src_true.extend(source_types)
                src_pred.extend(preds)
        return self._metrics(y_true, y_pred, src_true, src_pred)

    def evaluate(self, mode: str, data_root: str | Path, batch_size: int, num_workers: int) -> dict:
        data_root = Path(data_root)
        if mode == "seen":
            ds = DeepfakeDataset(data_root / "val", transform=RobustTransforms(self.image_size, training=False))
        elif mode == "unseen":
            ds = DeepfakeDataset(data_root / "test", transform=RobustTransforms(self.image_size, training=False))
        elif mode == "degraded":
            ds = DeepfakeDataset(data_root / "test", transform=build_degraded_transform(self.image_size))
        else:
            raise ValueError("mode must be one of: seen, unseen, degraded")

        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return self._predict_loader(loader)

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
