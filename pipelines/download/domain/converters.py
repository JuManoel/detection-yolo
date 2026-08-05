from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from typing import Protocol

from pipelines.download.domain.models import DatasetSpec, Sample
from pipelines.shared.domain.config import DEFAULT_TRAIN_CONFIG


class Converter(Protocol):
    """Converts a raw dataset directory into detection samples."""

    name: str

    def convert(self, raw_dir) -> list[Sample]:
        ...


def parse_dataset_list(text: str) -> list[DatasetSpec]:
    specs: list[DatasetSpec] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        specs.append(DatasetSpec(source=line, kind="kaggle"))
    specs.append(DatasetSpec(source="coco", kind="coco"))
    return specs


def split_samples(
    samples: Sequence[Sample],
    train_ratio: float = DEFAULT_TRAIN_CONFIG.train_ratio,
    seed: int = DEFAULT_TRAIN_CONFIG.seed,
) -> tuple[list[Sample], list[Sample]]:
    """Shuffle and split into train/val. Only samples with boxes are kept."""
    usable = [s for s in samples if s.has_boxes]
    rng = random.Random(seed)
    shuffled = list(usable)
    rng.shuffle(shuffled)
    if not shuffled:
        return [], []
    cut = max(1, int(len(shuffled) * train_ratio)) if len(shuffled) > 1 else 1
    if cut >= len(shuffled) and len(shuffled) > 1:
        cut = len(shuffled) - 1
    return shuffled[:cut], shuffled[cut:]


def merge_samples(groups: Iterable[Iterable[Sample]]) -> list[Sample]:
    merged: list[Sample] = []
    for group in groups:
        merged.extend(group)
    return merged
