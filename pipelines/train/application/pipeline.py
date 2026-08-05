from __future__ import annotations

import logging

from pipelines.shared.domain.config import DEFAULT_TRAIN_CONFIG, TrainConfig
from pipelines.shared.domain.paths import PathConfig
from pipelines.shared.infrastructure.fs import ensure_dir
from pipelines.shared.infrastructure.gpu import freest_device, is_cuda_oom
from pipelines.train.domain.train_spec import TrainJob, TrainResult
from pipelines.train.infrastructure.yolo_trainer import YoloTrainer

logger = logging.getLogger(__name__)


class TrainPipeline:
    """Train YOLO26 n/m/x with AdamW, early stopping, OOM batch halving."""

    def __init__(
        self,
        paths: PathConfig | None = None,
        config: TrainConfig = DEFAULT_TRAIN_CONFIG,
        trainer: YoloTrainer | None = None,
    ) -> None:
        self.paths = paths or PathConfig.from_cwd()
        self.config = config
        self.trainer = trainer or YoloTrainer()

    def run(self) -> list[TrainResult]:
        data_yaml = self.paths.yolo_yaml
        if not data_yaml.exists():
            raise FileNotFoundError(
                f"Missing {data_yaml}. Run the download pipeline first (`uv run download`)."
            )
        ensure_dir(self.paths.detect_dir)
        device = freest_device()
        results: list[TrainResult] = []

        for model in self.config.models:
            result = self._train_with_oom_retry(model, data_yaml, device)
            results.append(result)
        return results

    def _train_with_oom_retry(self, model, data_yaml, device) -> TrainResult:
        batch = model.batch
        last_error: str | None = None
        while batch >= 1:
            job = TrainJob.from_config(
                model=model,
                data_yaml=data_yaml,
                project=self.paths.detect_dir,
                config=self.config,
                device=device,
                batch=batch,
            )
            try:
                best = self.trainer.train(job)
                return TrainResult(
                    model_stem=model.stem,
                    best_weights=best,
                    final_batch=batch,
                    success=True,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if is_cuda_oom(exc) and batch > 1:
                    new_batch = max(1, batch // 2)
                    logger.warning(
                        "OOM on %s with batch=%d — retrying with batch=%d",
                        model.stem,
                        batch,
                        new_batch,
                    )
                    batch = new_batch
                    continue
                logger.exception("Training failed for %s", model.stem)
                return TrainResult(
                    model_stem=model.stem,
                    best_weights=None,
                    final_batch=batch,
                    success=False,
                    error=last_error,
                )
        return TrainResult(
            model_stem=model.stem,
            best_weights=None,
            final_batch=batch,
            success=False,
            error=last_error or "batch reduced below 1",
        )
