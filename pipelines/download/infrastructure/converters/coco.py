from __future__ import annotations

from pathlib import Path

from pipelines.download.domain.models import Sample
from pipelines.download.infrastructure.coco_client import CocoBirdClient


class CocoConverter:
    name = "coco-birds"

    def convert(self, raw_dir: Path) -> list[Sample]:
        return CocoBirdClient().to_samples(raw_dir)
