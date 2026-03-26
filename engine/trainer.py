from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from models.main_model import MultiSignalDeepfakeDetector


class Trainer:
    def __init__(self, config: dict, device: torch.device) -> None:
        self.cfg = config
        self.device = device
        self.model = MultiSignalDeepfakeDetector().to(device)
        self.criterion = nn.BCEWithLogitsLoss()

    def _run_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer | None) -> tuple[float, float]:
        is_train = optimizer is not None
        self.model.train(is_train)
        total_loss = 0.0
        correct = 0
        total = 0

        context = torch.enable_grad() if is_train else torch.no_grad()
        with context:
            for images, labels, _ in loader:
                images = images.to(self.device)
                labels = labels.float().to(self.device)

                if is_train:
                    optimizer.zero_grad(set_to_none=True)

                out = self.model(images)
                logits = out["logits"]
                loss = self.criterion(logits, labels)

                if is_train:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * images.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).long()
                correct += (preds == labels.long()).sum().item()
                total += images.size(0)

        return total_loss / max(1, total), correct / max(1, total)

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Path:
        ckpt_dir = Path(self.cfg["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_path = ckpt_dir / "best_multisignal.pt"

        self.model.freeze_vit()
        opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.cfg["phase1_lr"], weight_decay=self.cfg["weight_decay"])

        best_acc = -1.0
        for epoch in range(1, self.cfg["phase1_epochs"] + 1):
            tr_loss, tr_acc = self._run_epoch(train_loader, opt)
            va_loss, va_acc = self._run_epoch(val_loader, None)
            print(f"[Phase1][{epoch}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f}")
            if va_acc > best_acc:
                best_acc = va_acc
                torch.save({"model": self.model.state_dict(), "config": self.cfg}, best_path)

        self.model.unfreeze_vit_last_layers(self.cfg["vit_unfreeze_layers"])
        opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.cfg["phase2_lr"], weight_decay=self.cfg["weight_decay"])

        for epoch in range(1, self.cfg["phase2_epochs"] + 1):
            tr_loss, tr_acc = self._run_epoch(train_loader, opt)
            va_loss, va_acc = self._run_epoch(val_loader, None)
            print(f"[Phase2][{epoch}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f}")
            if va_acc > best_acc:
                best_acc = va_acc
                torch.save({"model": self.model.state_dict(), "config": self.cfg}, best_path)

        print(f"Best checkpoint: {best_path} val_acc={best_acc:.4f}")
        return best_path
