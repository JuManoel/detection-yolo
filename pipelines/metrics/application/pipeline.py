from __future__ import annotations

import logging
from pathlib import Path

from pipelines.metrics.domain.metrics_spec import MetricReport
from pipelines.metrics.infrastructure.report_writer import ReportWriter
from pipelines.metrics.infrastructure.yolo_validator import YoloValidator
from pipelines.shared.domain.config import DEFAULT_TRAIN_CONFIG, TrainConfig
from pipelines.shared.domain.paths import PathConfig
from pipelines.shared.infrastructure.gpu import freest_device

logger = logging.getLogger(__name__)


class MetricsPipeline:
    """Evaluate trained models on the unified val split."""

    def __init__(
        self,
        paths: PathConfig | None = None,
        config: TrainConfig = DEFAULT_TRAIN_CONFIG,
        validator: YoloValidator | None = None,
        writer: ReportWriter | None = None,
    ) -> None:
        self.paths = paths or PathConfig.from_cwd()
        self.config = config
        self.validator = validator or YoloValidator()
        self.writer = writer or ReportWriter()

    def run(self) -> list[MetricReport]:
        data_yaml = self.paths.yolo_yaml
        if not data_yaml.exists():
            raise FileNotFoundError(f"Missing {data_yaml}. Run download first.")

        val_images = self._count_val_images()
        device = freest_device()
        reports: list[MetricReport] = []

        for model in self.config.models:
            weights = self._find_best(model.stem)
            if weights is None:
                logger.warning("No best.pt for %s — skip", model.stem)
                continue
            report = self.validator.validate(
                weights,
                data_yaml,
                model_stem=model.stem,
                val_images=val_images,
                device=device,
            )
            reports.append(report)

        if not reports:
            raise RuntimeError("No trained weights found under runs/detect/")

        self.writer.write(self.paths.metrics_dir, reports)
        return reports

    def _count_val_images(self) -> int:
        val_dir = self.paths.yolo_dir / "images" / "val"
        if not val_dir.is_dir():
            return 0
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return sum(1 for p in val_dir.iterdir() if p.suffix.lower() in exts)

    def _find_best(self, model_stem: str) -> Path | None:
        direct = self.paths.detect_dir / model_stem / "weights" / "best.pt"
        if direct.exists():
            return direct
        matches = list(self.paths.detect_dir.rglob(f"*/{model_stem}/**/best.pt"))
        if not matches:
            matches = [
                p
                for p in self.paths.detect_dir.rglob("best.pt")
                if model_stem in str(p)
            ]
        return matches[0] if matches else None
