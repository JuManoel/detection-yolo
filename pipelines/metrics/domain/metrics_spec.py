from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MetricReport:
    model_stem: str
    weights: str
    val_images: int
    precision: float | None
    recall: float | None
    map50: float | None
    map50_95: float | None
    box_loss: float | None = None
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data
