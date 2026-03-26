from __future__ import annotations

import random

import torch
import torch.nn as nn

from models.dwt_branch import DWTBranch
from models.fft_branch import FFTBranch
from models.fusion import FeatureFusion
from models.physical_features import PhysicalFeatures
from models.srm_branch import SRMBranch
from models.vit_branch import ViTBranch


class MultiSignalDeepfakeDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.srm_branch = SRMBranch(out_dim=256)
        self.fft_branch = FFTBranch(out_dim=128)
        self.dwt_branch = DWTBranch(out_dim=128)
        self.vit_branch = ViTBranch(out_dim=256)
        self.physical_features = PhysicalFeatures()
        self.norm_srm = nn.LayerNorm(256)
        self.norm_fft = nn.LayerNorm(128)
        self.norm_dwt = nn.LayerNorm(128)
        self.norm_vit = nn.LayerNorm(256)
        self.norm_phys = nn.LayerNorm(5)
        self.branch_dropout = nn.Dropout(p=0.1)
        self.branch_gates = nn.Parameter(torch.ones(5))
        self.branch_drop_prob = 0.15
        self.fusion = FeatureFusion(in_dim=256 + 128 + 128 + 256 + 5, hidden_dim=256, dropout=0.3)

    def freeze_vit(self) -> None:
        self.vit_branch.freeze()

    def unfreeze_vit_last_layers(self, num_layers: int = 2) -> None:
        self.vit_branch.unfreeze_last_layers(num_layers=num_layers)

    def _apply_ablation(self, branch_name: str | None, feats: dict[str, torch.Tensor]) -> None:
        if branch_name in feats:
            feats[branch_name] = torch.zeros_like(feats[branch_name])

    def _apply_training_branch_drop(self, feats: dict[str, torch.Tensor]) -> None:
        if not self.training or random.random() > self.branch_drop_prob:
            return
        drop_key = random.choice(list(feats.keys()))
        feats[drop_key] = torch.zeros_like(feats[drop_key])

    def forward(
        self,
        x: torch.Tensor,
        return_branch_outputs: bool = False,
        ablate_branch: str | None = None,
    ) -> dict[str, torch.Tensor]:
        srm = self.srm_branch(x)
        fft = self.fft_branch(x)
        dwt = self.dwt_branch(x)
        vit = self.vit_branch(x)
        phys = self.physical_features(x)

        srm = self.branch_dropout(self.norm_srm(srm))
        fft = self.branch_dropout(self.norm_fft(fft))
        dwt = self.branch_dropout(self.norm_dwt(dwt))
        vit = self.branch_dropout(self.norm_vit(vit))
        phys = self.branch_dropout(self.norm_phys(phys))

        feats = {"srm": srm, "fft": fft, "dwt": dwt, "vit": vit, "physical": phys}
        self._apply_training_branch_drop(feats)
        self._apply_ablation(ablate_branch, feats)

        gate_weights = torch.softmax(self.branch_gates, dim=0)
        weighted_feats = {
            "srm": feats["srm"] * gate_weights[0],
            "fft": feats["fft"] * gate_weights[1],
            "dwt": feats["dwt"] * gate_weights[2],
            "vit": feats["vit"] * gate_weights[3],
            "physical": feats["physical"] * gate_weights[4],
        }

        fused = torch.cat(
            [weighted_feats["srm"], weighted_feats["fft"], weighted_feats["dwt"], weighted_feats["vit"], weighted_feats["physical"]],
            dim=1,
        )
        logits = self.fusion(fused)
        prob = torch.sigmoid(logits)

        out = {"logits": logits, "probability": prob, "gate_weights": gate_weights}
        if return_branch_outputs:
            out["branches"] = weighted_feats
        return out
