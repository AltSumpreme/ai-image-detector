from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class DatasetConfig:
    root_dir: Path = Path("dataset")
    real_dir_name: str = "real"
    ai_dir_name: str = "ai"
    image_size: int = 224
    val_split: float = 0.2
    random_seed: int = 42


@dataclass
class TrainingConfig:
    model_name: str = "efficientnet_b0"
    num_classes: int = 2
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 12
    checkpoint_dir: Path = Path("checkpoints")
    best_model_name: str = "best_efficientnet.pt"
    log_every_n_steps: int = 20
    pretrained: bool = True


@dataclass
class InferenceConfig:
    model_path: Path = Path("checkpoints/best_efficientnet.pt")
    confidence_threshold: float = 0.5


@dataclass
class FeatureConfig:
    enabled_extractors: List[str] = field(default_factory=lambda: ["rgb"])
    fft_bins: int = 32


@dataclass
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def ensure_dirs(config: AppConfig) -> None:
    config.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)
