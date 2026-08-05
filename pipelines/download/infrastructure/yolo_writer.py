from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image
import yaml

from pipelines.download.domain.models import Sample
from pipelines.shared.domain.config import DEFAULT_TRAIN_CONFIG
from pipelines.shared.infrastructure.fs import clear_dir, copy_or_link, ensure_dir

logger = logging.getLogger(__name__)


class YoloWriter:
    """Write unified YOLO dataset (images/labels + data.yaml)."""

    def write(
        self,
        yolo_dir: Path,
        train: list[Sample],
        val: list[Sample],
        *,
        class_name: str = DEFAULT_TRAIN_CONFIG.class_name,
    ) -> Path:
        clear_dir(yolo_dir)
        for split_name, samples in (("train", train), ("val", val)):
            img_dir = ensure_dir(yolo_dir / "images" / split_name)
            lbl_dir = ensure_dir(yolo_dir / "labels" / split_name)
            for idx, sample in enumerate(samples):
                stem = f"{sample.source}_{idx:06d}_{sample.image_path.stem}"
                # sanitize stem
                stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
                ext = sample.image_path.suffix.lower() or ".jpg"
                dest_img = img_dir / f"{stem}{ext}"
                dest_lbl = lbl_dir / f"{stem}.txt"
                copy_or_link(sample.image_path, dest_img)
                dest_lbl.write_text(
                    "\n".join(box.to_line() for box in sample.boxes) + "\n",
                    encoding="utf-8",
                )

        yaml_path = yolo_dir / "data.yaml"
        payload = {
            "path": str(yolo_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "nc": 1,
            "names": {0: class_name},
        }
        yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        logger.info(
            "Wrote YOLO dataset: train=%d val=%d → %s",
            len(train),
            len(val),
            yolo_dir,
        )
        return yaml_path


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size  # (w, h)
