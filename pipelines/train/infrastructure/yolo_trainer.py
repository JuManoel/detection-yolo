from __future__ import annotations

import logging
from pathlib import Path

from pipelines.train.domain.train_spec import TrainJob

logger = logging.getLogger(__name__)


class YoloTrainer:
    """Thin Ultralytics adapter."""

    def train(self, job: TrainJob) -> Path:
        from ultralytics import YOLO

        logger.info(
            "Training %s epochs=%d batch=%d device=%s optimizer=%s",
            job.model.weights,
            job.epochs,
            job.batch,
            job.device,
            job.optimizer,
        )
        model = YOLO(job.model.weights)
        model.train(
            data=str(job.data_yaml),
            epochs=job.epochs,
            patience=job.patience,
            optimizer=job.optimizer,
            imgsz=job.imgsz,
            batch=job.batch,
            device=job.device,
            project=str(job.project),
            name=job.name,
            seed=job.seed,
            exist_ok=True,
            pretrained=True,
            single_cls=True,
        )
        best = job.project / job.name / "weights" / "best.pt"
        if not best.exists():
            # Ultralytics sometimes nests runs
            candidates = list((job.project / job.name).rglob("best.pt"))
            if candidates:
                best = candidates[0]
        if not best.exists():
            raise FileNotFoundError(f"best.pt not found after training {job.model.weights}")
        return best
