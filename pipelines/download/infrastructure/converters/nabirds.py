from __future__ import annotations

import logging
from pathlib import Path

from pipelines.download.domain.models import Sample
from pipelines.download.infrastructure.yolo_writer import image_size
from pipelines.shared.domain.bbox import xywh_pixels_to_yolo
from pipelines.shared.domain.config import DEFAULT_TRAIN_CONFIG

logger = logging.getLogger(__name__)


def _find_nabirds_root(raw_dir: Path) -> Path | None:
    hits: list[Path] = []
    for images_txt in raw_dir.rglob("images.txt"):
        parent = images_txt.parent
        if (parent / "bounding_boxes.txt").exists() and (parent / "images").is_dir():
            hits.append(parent)
    if not hits:
        return None
    for hit in hits:
        name = str(hit).lower()
        if "nabird" in name:
            return hit
    # NABirds image ids are UUIDs; CUB uses numeric ids
    for hit in hits:
        first = (hit / "images.txt").read_text(encoding="utf-8", errors="ignore").splitlines()[:1]
        if first and len(first[0].split()[0]) > 8 and not first[0].split()[0].isdigit():
            return hit
    return hits[0]


class NabirdsConverter:
    name = "nabirds"

    def convert(self, raw_dir: Path) -> list[Sample]:
        root = _find_nabirds_root(raw_dir)
        if root is None:
            logger.warning("NABirds structure not found under %s", raw_dir)
            return []

        images: dict[str, str] = {}
        with (root / "images.txt").open() as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    images[parts[0]] = parts[1]

        boxes_raw: dict[str, tuple[float, float, float, float]] = {}
        with (root / "bounding_boxes.txt").open() as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    boxes_raw[parts[0]] = tuple(map(float, parts[1:5]))  # type: ignore[assignment]

        images_dir = root / "images"
        samples: list[Sample] = []
        class_id = DEFAULT_TRAIN_CONFIG.class_id
        for img_id, rel in images.items():
            if img_id not in boxes_raw:
                continue
            path = images_dir / rel
            if not path.exists():
                # some mirrors store without nested folders matching path
                alt = images_dir / Path(rel).name
                path = alt if alt.exists() else path
            if not path.exists():
                continue
            try:
                w, h = image_size(path)
            except OSError:
                continue
            x, y, bw, bh = boxes_raw[img_id]
            box = xywh_pixels_to_yolo(x, y, bw, bh, w, h, class_id=class_id)
            samples.append(Sample(image_path=path, boxes=(box,), source=self.name))
        logger.info("NABirds converter: %d samples from %s", len(samples), root)
        return samples
