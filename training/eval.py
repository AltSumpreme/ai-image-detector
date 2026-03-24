from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import ImageAIDetectionDataset
from data.transforms import get_eval_transforms
from models.efficientnet import build_efficientnet
from utils.metrics import compute_confusion_matrix, compute_metrics


LABELS = ["REAL", "AI_GENERATED"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained EfficientNet checkpoint")
    parser.add_argument("--dataset-root", type=str, default="dataset")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_efficientnet(
        model_name=checkpoint["model_name"],
        num_classes=checkpoint["num_classes"],
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    dataset = ImageAIDetectionDataset(args.dataset_root, transform=get_eval_transforms(checkpoint.get("image_size", 224)))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    metrics = compute_metrics(y_true_arr, y_pred_arr)
    cm = compute_confusion_matrix(y_true_arr, y_pred_arr)

    print("=== Evaluation Results ===")
    print(f"Accuracy : {metrics.accuracy:.4f}")
    print(f"Precision: {metrics.precision:.4f}")
    print(f"Recall   : {metrics.recall:.4f}")
    print("\nConfusion Matrix [rows=true, cols=pred]")
    print(f"Labels: {LABELS}")
    print(cm)


if __name__ == "__main__":
    main()
