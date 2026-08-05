from __future__ import annotations

import logging
from pathlib import Path

from pipelines.download.domain.models import Sample
from pipelines.download.infrastructure.yolo_writer import image_size
from pipelines.shared.domain.bbox import xywh_pixels_to_yolo
from pipelines.shared.domain.config import DEFAULT_TRAIN_CONFIG

logger = logging.getLogger(__name__)


def _find_cub_root(raw_dir: Path) -> Path | None:
    """Locate CUB root; prefer paths that look like CUB_200_2011."""
    hits: list[Path] = []
    for images_txt in raw_dir.rglob("images.txt"):
        parent = images_txt.parent
        if (parent / "bounding_boxes.txt").exists() and (parent / "images").is_dir():
            hits.append(parent)
    if not hits:
        return None
    for hit in hits:
        if "cub" in hit.name.lower() or "cub" in str(hit).lower():
            return hit
    # Ambiguous (e.g. NABirds-like): only accept if classes.txt has 200-style names
    for hit in hits:
        classes = hit / "classes.txt"
        if classes.exists():
            lines = [ln for ln in classes.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
            if len(lines) == 200:
                return hit
    return None


class CubConverter:
    name = "cub2002011"

    def convert(self, raw_dir: Path) -> list[Sample]:
        root = _find_cub_root(raw_dir)
        if root is None:
            logger.warning("CUB structure not found under %s", raw_dir)
            return []

        images: dict[str, str] = {}
        with (root / "images.txt").open() as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
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
                continue
            try:
                w, h = image_size(path)
            except OSError:
                continue
            x, y, bw, bh = boxes_raw[img_id]
            box = xywh_pixels_to_yolo(x, y, bw, bh, w, h, class_id=class_id)
            samples.append(Sample(image_path=path, boxes=(box,), source=self.name))
        logger.info("CUB converter: %d samples from %s", len(samples), root)
        return samples
