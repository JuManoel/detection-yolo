from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    """YOLO26 checkpoint and default batch size."""

    weights: str
    batch: int

    @property
    def stem(self) -> str:
        return self.weights.removesuffix(".pt")


@dataclass(frozen=True)
class TrainConfig:
    """Default fine-tuning hyperparameters."""

    epochs: int = 100
    patience: int = 10
    optimizer: str = "AdamW"
    imgsz: int = 640
    seed: int = 42
    train_ratio: float = 0.8
    class_name: str = "bird"
    class_id: int = 0
    models: tuple[ModelSpec, ...] = field(
        default_factory=lambda: (
            ModelSpec("yolo26n.pt", batch=32),
            ModelSpec("yolo26m.pt", batch=16),
            ModelSpec("yolo26x.pt", batch=8),
        )
    )


DEFAULT_TRAIN_CONFIG = TrainConfig()
