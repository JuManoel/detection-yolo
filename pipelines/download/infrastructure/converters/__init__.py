from __future__ import annotations

from pathlib import Path

from pipelines.download.domain.models import Sample
from pipelines.download.infrastructure.converters.auto import AutoConverter
from pipelines.download.infrastructure.converters.coco import CocoConverter
from pipelines.download.infrastructure.converters.cub import CubConverter
from pipelines.download.infrastructure.converters.fbd import FbdConverter
from pipelines.download.infrastructure.converters.nabirds import NabirdsConverter


def convert_raw(slug: str, raw_dir: Path) -> list[Sample]:
    """Pick converter by dataset slug."""
    mapping = {
        "cub2002011": CubConverter(),
        "nabirds": NabirdsConverter(),
        "fbd-sv-2024": FbdConverter(),
        "coco-birds": CocoConverter(),
        "bird-dataset": AutoConverter(),
    }
    converter = mapping.get(slug, AutoConverter())
    return converter.convert(raw_dir)
