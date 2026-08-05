from __future__ import annotations

import json
import logging
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlretrieve

from pipelines.shared.domain.bbox import xywh_pixels_to_yolo
from pipelines.shared.domain.config import DEFAULT_TRAIN_CONFIG
from pipelines.shared.infrastructure.fs import ensure_dir
from pipelines.download.domain.models import Sample

logger = logging.getLogger(__name__)

COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMG_ROOT = "http://images.cocodataset.org/zips"
# Official COCO category id for bird
COCO_BIRD_CATEGORY_ID = 16


class CocoBirdClient:
    """Download COCO annotations and only images that contain birds."""

    def __init__(self, max_workers: int = 8) -> None:
        self.max_workers = max_workers

    def download(self, raw_dir: Path) -> Path:
        target = ensure_dir(raw_dir / "coco-birds")
        marker = target / ".download_complete"
        if marker.exists() and (target / "annotations").exists():
            logger.info("Skipping existing COCO bird subset at %s", target)
            return target

        ann_dir = ensure_dir(target / "annotations")
        zip_path = target / "annotations_trainval2017.zip"
        if not (ann_dir / "instances_train2017.json").exists():
            logger.info("Downloading COCO annotations…")
            urlretrieve(COCO_ANN_URL, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(target)
            zip_path.unlink(missing_ok=True)

        for split in ("train2017", "val2017"):
            self._download_bird_images(target, split)

        marker.touch()
        return target

    def _download_bird_images(self, target: Path, split: str) -> None:
        ann_path = target / "annotations" / f"instances_{split}.json"
        with ann_path.open() as f:
            data = json.load(f)

        bird_img_ids = {
            a["image_id"]
            for a in data["annotations"]
            if a["category_id"] == COCO_BIRD_CATEGORY_ID
        }
        images = {img["id"]: img for img in data["images"] if img["id"] in bird_img_ids}
        img_dir = ensure_dir(target / "images" / split)

        pending = []
        for img in images.values():
            dest = img_dir / img["file_name"]
            if not dest.exists():
                url = f"http://images.cocodataset.org/{split}/{img['file_name']}"
                pending.append((url, dest))

        if not pending:
            logger.info("COCO %s bird images already present (%d)", split, len(images))
            return

        logger.info("Downloading %d COCO bird images for %s…", len(pending), split)

        def _fetch(pair: tuple[str, Path]) -> None:
            url, dest = pair
            try:
                urlretrieve(url, dest)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed %s: %s", url, exc)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [pool.submit(_fetch, p) for p in pending]
            for fut in as_completed(futures):
                fut.result()

    def to_samples(self, raw_dir: Path) -> list[Sample]:
        """Convert downloaded COCO bird subset into Samples."""
        samples: list[Sample] = []
        class_id = DEFAULT_TRAIN_CONFIG.class_id
        for split in ("train2017", "val2017"):
            ann_path = raw_dir / "annotations" / f"instances_{split}.json"
            if not ann_path.exists():
                continue
            with ann_path.open() as f:
                data = json.load(f)
            images = {img["id"]: img for img in data["images"]}
            by_image: dict[int, list] = {}
            for ann in data["annotations"]:
                if ann["category_id"] != COCO_BIRD_CATEGORY_ID:
                    continue
                by_image.setdefault(ann["image_id"], []).append(ann)

            for image_id, anns in by_image.items():
                info = images.get(image_id)
                if not info:
                    continue
                img_path = raw_dir / "images" / split / info["file_name"]
                if not img_path.exists():
                    continue
                boxes = []
                for ann in anns:
                    x, y, w, h = ann["bbox"]
                    boxes.append(
                        xywh_pixels_to_yolo(
                            x, y, w, h, info["width"], info["height"], class_id=class_id
                        )
                    )
                if boxes:
                    samples.append(
                        Sample(
                            image_path=img_path,
                            boxes=tuple(boxes),
                            source="coco-birds",
                        )
                    )
        return samples
