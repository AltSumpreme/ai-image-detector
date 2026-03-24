from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from data.dataset import ImageAIDetectionDataset, build_train_val_indices
from data.transforms import get_eval_transforms, get_train_transforms
from models.efficientnet import build_efficientnet
from utils.config import AppConfig, ensure_dirs
from utils.metrics import MetricBundle, compute_metrics


@dataclass
class EpochResult:
    loss: float
    metrics: MetricBundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training pipeline for REAL vs AI_GENERATED classification")
    parser.add_argument("--dataset-root", type=str, default="dataset", help="Root dataset directory containing real/ and ai/")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--checkpoint-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(config: AppConfig) -> Tuple[DataLoader, DataLoader]:
    split_dataset = ImageAIDetectionDataset(root_dir=config.dataset.root_dir, transform=None)
    train_split, val_split = build_train_val_indices(split_dataset, config.dataset.val_split, config.dataset.random_seed)

    train_source = ImageAIDetectionDataset(
        root_dir=config.dataset.root_dir,
        transform=get_train_transforms(config.dataset.image_size),
    )
    val_source = ImageAIDetectionDataset(
        root_dir=config.dataset.root_dir,
        transform=get_eval_transforms(config.dataset.image_size),
    )

    train_dataset = Subset(train_source, train_split.indices)
    val_dataset = Subset(val_source, val_split.indices)

    print(
        "Dataset summary | "
        f"total={len(split_dataset)} "
        f"train={len(train_dataset)} "
        f"val={len(val_dataset)} "
        f"class_counts={split_dataset.class_counts()}"
    )

    loader_kwargs: Dict[str, object] = {
        "batch_size": config.training.batch_size,
        "num_workers": config.training.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochResult:
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())

    loss_avg = running_loss / max(1, len(loader))
    metrics = compute_metrics(np.array(y_true), np.array(y_pred))
    return EpochResult(loss=loss_avg, metrics=metrics)


def save_best_checkpoint(
    model: nn.Module,
    config: AppConfig,
    checkpoint_path: Path,
    best_epoch: int,
    best_val: EpochResult,
    history: list[dict],
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_name": config.training.model_name,
            "num_classes": config.training.num_classes,
            "image_size": config.dataset.image_size,
            "class_to_idx": {"REAL": 0, "AI_GENERATED": 1},
            "best_epoch": best_epoch,
            "val_metrics": best_val.metrics.as_dict(),
            "val_loss": best_val.loss,
            "history": history,
        },
        checkpoint_path,
    )


def main() -> None:
    args = parse_args()
    config = AppConfig()

    config.dataset.root_dir = Path(args.dataset_root)
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.num_workers is not None:
        config.training.num_workers = args.num_workers
    if args.checkpoint_name is not None:
        config.training.best_model_name = args.checkpoint_name
    if args.seed is not None:
        config.dataset.random_seed = args.seed

    set_global_seed(config.dataset.random_seed)
    ensure_dirs(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_dataloaders(config)

    model = build_efficientnet(
        model_name=config.training.model_name,
        num_classes=config.training.num_classes,
        pretrained=config.training.pretrained,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    best_state = None
    best_epoch = -1
    best_val_result: EpochResult | None = None
    history: list[dict] = []

    for epoch in range(1, config.training.epochs + 1):
        train_result = run_epoch(model, train_loader, criterion, device, optimizer)
        val_result = run_epoch(model, val_loader, criterion, device, optimizer=None)

        epoch_row = {
            "epoch": epoch,
            "train_loss": train_result.loss,
            "val_loss": val_result.loss,
            "train_accuracy": train_result.metrics.accuracy,
            "train_precision": train_result.metrics.precision,
            "train_recall": train_result.metrics.recall,
            "val_accuracy": val_result.metrics.accuracy,
            "val_precision": val_result.metrics.precision,
            "val_recall": val_result.metrics.recall,
        }
        history.append(epoch_row)

        print(
            f"Epoch {epoch:03d}/{config.training.epochs:03d} | "
            f"train_loss={train_result.loss:.4f} val_loss={val_result.loss:.4f} | "
            f"train_acc={train_result.metrics.accuracy:.4f} val_acc={val_result.metrics.accuracy:.4f} | "
            f"val_precision={val_result.metrics.precision:.4f} val_recall={val_result.metrics.recall:.4f}"
        )

        if best_val_result is None or val_result.metrics.accuracy > best_val_result.metrics.accuracy:
            best_epoch = epoch
            best_val_result = val_result
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None or best_val_result is None:
        raise RuntimeError("Training completed without producing a valid checkpoint state")

    model.load_state_dict(best_state)
    checkpoint_path = config.training.checkpoint_dir / config.training.best_model_name
    save_best_checkpoint(model, config, checkpoint_path, best_epoch, best_val_result, history)

    history_path = checkpoint_path.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2))

    print(
        f"Saved best checkpoint: {checkpoint_path} | "
        f"best_epoch={best_epoch} val_acc={best_val_result.metrics.accuracy:.4f}"
    )
    print(f"Saved metric history: {history_path}")


if __name__ == "__main__":
    main()
