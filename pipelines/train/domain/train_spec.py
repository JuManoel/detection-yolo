from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pipelines.shared.domain.config import ModelSpec, TrainConfig


@dataclass(frozen=True)
class TrainJob:
    """One model fine-tuning job."""

    model: ModelSpec
    data_yaml: Path
    project: Path
    name: str
    epochs: int
    patience: int
    optimizer: str
    imgsz: int
    batch: int
    device: str | int
    seed: int

    @classmethod
    def from_config(
        cls,
        model: ModelSpec,
        data_yaml: Path,
        project: Path,
        config: TrainConfig,
        device: str | int,
        batch: int | None = None,
    ) -> TrainJob:
        return cls(
            model=model,
            data_yaml=data_yaml,
            project=project,
            name=model.stem,
            epochs=config.epochs,
            patience=config.patience,
            optimizer=config.optimizer,
            imgsz=config.imgsz,
            batch=batch if batch is not None else model.batch,
            device=device,
            seed=config.seed,
        )


@dataclass(frozen=True)
class TrainResult:
    model_stem: str
    best_weights: Path | None
    final_batch: int
    success: bool
    error: str | None = None
