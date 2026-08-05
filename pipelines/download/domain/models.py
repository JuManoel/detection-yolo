from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pipelines.shared.domain.bbox import YoloBox


@dataclass(frozen=True)
class DatasetSpec:
    """Kaggle dataset slug or special source name."""

    source: str
    kind: str = "kaggle"  # kaggle | coco

    @property
    def slug(self) -> str:
        if self.kind == "coco":
            return "coco-birds"
        return self.source.strip().split("/")[-1]


@dataclass(frozen=True)
class Sample:
    """One image with YOLO boxes (class 0 = bird)."""

    image_path: Path
    boxes: tuple[YoloBox, ...]
    source: str = ""

    @property
    def has_boxes(self) -> bool:
        return len(self.boxes) > 0
