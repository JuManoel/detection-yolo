from __future__ import annotations

import logging
from pathlib import Path

from pipelines.download.domain.converters import merge_samples, parse_dataset_list, split_samples
from pipelines.download.infrastructure.coco_client import CocoBirdClient
from pipelines.download.infrastructure.converters import convert_raw
from pipelines.download.infrastructure.kaggle_client import KaggleClient
from pipelines.download.infrastructure.yolo_writer import YoloWriter
from pipelines.shared.domain.config import DEFAULT_TRAIN_CONFIG, TrainConfig
from pipelines.shared.domain.paths import PathConfig
from pipelines.shared.infrastructure.fs import ensure_dir

logger = logging.getLogger(__name__)


class DownloadPipeline:
    """Orchestrate: download raw → convert → merge/split → write YOLO."""

    def __init__(
        self,
        paths: PathConfig | None = None,
        config: TrainConfig = DEFAULT_TRAIN_CONFIG,
        kaggle: KaggleClient | None = None,
        coco: CocoBirdClient | None = None,
        writer: YoloWriter | None = None,
    ) -> None:
        self.paths = paths or PathConfig.from_cwd()
        self.config = config
        self.kaggle = kaggle or KaggleClient()
        self.coco = coco or CocoBirdClient()
        self.writer = writer or YoloWriter()

    def run(self) -> Path:
        ensure_dir(self.paths.raw_dir)
        text = self.paths.datasets_list.read_text(encoding="utf-8")
        specs = parse_dataset_list(text)
        logger.info("Dataset specs: %s", [s.source for s in specs])

        groups = []
        for spec in specs:
            if spec.kind == "coco":
                raw = self.coco.download(self.paths.raw_dir)
            else:
                raw = self.kaggle.download(spec.source, self.paths.raw_dir)
            samples = convert_raw(spec.slug, raw)
            logger.info("%s → %d samples with boxes", spec.slug, len(samples))
            groups.append(samples)

        merged = merge_samples(groups)
        train, val = split_samples(
            merged,
            train_ratio=self.config.train_ratio,
            seed=self.config.seed,
        )
        if not train or not val:
            raise RuntimeError(
                f"Not enough labeled samples after conversion (train={len(train)}, val={len(val)})"
            )

        yaml_path = self.writer.write(
            self.paths.yolo_dir,
            train,
            val,
            class_name=self.config.class_name,
        )
        logger.info("Download pipeline complete: %s", yaml_path)
        return yaml_path
