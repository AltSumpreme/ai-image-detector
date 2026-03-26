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
        self.criterion: nn.Module | None = None

    def _build_criterion(self, train_loader: DataLoader) -> nn.Module:
        if self.cfg.get("use_pos_weight", True):
            dataset = train_loader.dataset
            labels = [s.label for s in dataset.samples] if hasattr(dataset, "samples") else []
            pos = max(1, sum(1 for y in labels if y == 1))
            neg = max(1, sum(1 for y in labels if y == 0))
            pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=self.device)
            print(f"Using BCEWithLogitsLoss(pos_weight={pos_weight.item():.4f})")
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return nn.BCEWithLogitsLoss()

    def _run_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer | None) -> tuple[float, float, dict[str, float]]:
        is_train = optimizer is not None
        self.model.train(is_train)
        total_loss = 0.0
        correct = 0
        total = 0
        branch_norm_sums = {"srm": 0.0, "fft": 0.0, "dwt": 0.0, "vit": 0.0, "physical": 0.0}
        gate_sum = torch.zeros(5, device=self.device)
        num_batches = 0

        context = torch.enable_grad() if is_train else torch.no_grad()
        with context:
            for images, labels, _ in loader:
                images = images.to(self.device)
                labels = labels.float().to(self.device)

                if is_train:
                    optimizer.zero_grad(set_to_none=True)

                out = self.model(images, return_branch_outputs=True)
                logits = out["logits"]
                if self.criterion is None:
                    raise RuntimeError("Loss criterion is not initialized")
                loss = self.criterion(logits, labels)

                if is_train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                total_loss += loss.item() * images.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).long()
                correct += (preds == labels.long()).sum().item()
                total += images.size(0)
                num_batches += 1

                branches = out["branches"]
                for key in branch_norm_sums:
                    branch_norm_sums[key] += branches[key].norm(dim=1).mean().item()
                gate_sum += out["gate_weights"]

        stats = {
            "srm_norm": branch_norm_sums["srm"] / max(1, num_batches),
            "fft_norm": branch_norm_sums["fft"] / max(1, num_batches),
            "dwt_norm": branch_norm_sums["dwt"] / max(1, num_batches),
            "vit_norm": branch_norm_sums["vit"] / max(1, num_batches),
            "physical_norm": branch_norm_sums["physical"] / max(1, num_batches),
            "gate_srm": (gate_sum[0] / max(1, num_batches)).item(),
            "gate_fft": (gate_sum[1] / max(1, num_batches)).item(),
            "gate_dwt": (gate_sum[2] / max(1, num_batches)).item(),
            "gate_vit": (gate_sum[3] / max(1, num_batches)).item(),
            "gate_phys": (gate_sum[4] / max(1, num_batches)).item(),
        }
        return total_loss / max(1, total), correct / max(1, total), stats

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Path:
        ckpt_dir = Path(self.cfg["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_path = ckpt_dir / "best_multisignal.pt"
        self.criterion = self._build_criterion(train_loader)

        self.model.freeze_vit()
        opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.cfg["phase1_lr"], weight_decay=self.cfg["weight_decay"])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, self.cfg["phase1_epochs"]))

        best_acc = -1.0
        for epoch in range(1, self.cfg["phase1_epochs"] + 1):
            tr_loss, tr_acc, tr_stats = self._run_epoch(train_loader, opt)
            va_loss, va_acc, va_stats = self._run_epoch(val_loader, None)
            print(f"[Phase1][{epoch}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f}")
            print(f"  branch_train={tr_stats}")
            print(f"  branch_val={va_stats}")
            if va_acc > best_acc:
                best_acc = va_acc
                torch.save({"model": self.model.state_dict(), "config": self.cfg}, best_path)
            sch.step()

        self.model.unfreeze_vit_last_layers(self.cfg["vit_unfreeze_layers"])
        opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.cfg["phase2_lr"], weight_decay=self.cfg["weight_decay"])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, self.cfg["phase2_epochs"]))

        for epoch in range(1, self.cfg["phase2_epochs"] + 1):
            tr_loss, tr_acc, tr_stats = self._run_epoch(train_loader, opt)
            va_loss, va_acc, va_stats = self._run_epoch(val_loader, None)
            print(f"[Phase2][{epoch}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f}")
            print(f"  branch_train={tr_stats}")
            print(f"  branch_val={va_stats}")
            if va_acc > best_acc:
                best_acc = va_acc
                torch.save({"model": self.model.state_dict(), "config": self.cfg}, best_path)
            sch.step()

        print(f"Best checkpoint: {best_path} val_acc={best_acc:.4f}")
        return best_path
