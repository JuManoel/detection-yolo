from __future__ import annotations

import json
import logging
from pathlib import Path

from pipelines.download.domain.models import Sample
from pipelines.download.infrastructure.converters.cub import CubConverter
from pipelines.download.infrastructure.converters.fbd import FbdConverter
from pipelines.download.infrastructure.converters.nabirds import NabirdsConverter
from pipelines.download.infrastructure.yolo_writer import image_size
from pipelines.shared.domain.bbox import YoloBox, xywh_pixels_to_yolo
from pipelines.shared.domain.config import DEFAULT_TRAIN_CONFIG

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class AutoConverter:
    """Inspect raw folder and convert YOLO / VOC / CUB-like / COCO layouts."""

    name = "auto"

    def convert(self, raw_dir: Path) -> list[Sample]:
        # Prefer known structured formats first
        for converter in (CubConverter(), NabirdsConverter(), FbdConverter()):
            samples = converter.convert(raw_dir)
            if samples:
                logger.info("AutoConverter used %s (%d samples)", converter.name, len(samples))
                return samples

        yolo = self._from_yolo_txt(raw_dir)
        if yolo:
            logger.info("AutoConverter: YOLO txt labels (%d)", len(yolo))
            return yolo

        coco = self._from_coco_json(raw_dir)
        if coco:
            logger.info("AutoConverter: COCO json (%d)", len(coco))
            return coco

        voc = FbdConverter().convert(raw_dir)
        if voc:
            return voc

        logger.warning(
            "No bounding-box annotations found under %s — skipping classification-only data",
            raw_dir,
        )
        return []

    def _from_yolo_txt(self, raw_dir: Path) -> list[Sample]:
        label_files = [p for p in raw_dir.rglob("*.txt") if p.name not in {
            "classes.txt",
            "data.yaml",
            "images.txt",
            "bounding_boxes.txt",
            "train_test_split.txt",
            "image_class_labels.txt",
            "classes.txt",
        }]
        # Heuristic: YOLO label lines start with int and 4 floats
        samples: list[Sample] = []
        class_id = DEFAULT_TRAIN_CONFIG.class_id
        images = {
            p.stem: p
            for p in raw_dir.rglob("*")
            if p.suffix.lower() in IMAGE_EXTS and p.is_file()
        }
        for lbl in label_files:
            if lbl.stem not in images:
                continue
            lines = []
            try:
                text = lbl.read_text(encoding="utf-8", errors="ignore").strip()
            except OSError:
                continue
            if not text:
                continue
            ok = True
            boxes: list[YoloBox] = []
            for line in text.splitlines():
                parts = line.split()
                if len(parts) != 5:
                    ok = False
                    break
                try:
                    _cid, xc, yc, w, h = parts
                    float(xc)
                    float(yc)
                    float(w)
                    float(h)
                    # remap any class to bird
                    boxes.append(
                        YoloBox(class_id, float(xc), float(yc), float(w), float(h)).clamp()
                    )
                except ValueError:
                    ok = False
                    break
            if ok and boxes:
                samples.append(
                    Sample(image_path=images[lbl.stem], boxes=tuple(boxes), source=self.name)
                )
        return samples

    def _from_coco_json(self, raw_dir: Path) -> list[Sample]:
        samples: list[Sample] = []
        class_id = DEFAULT_TRAIN_CONFIG.class_id
        for json_path in raw_dir.rglob("*.json"):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict) or "annotations" not in data or "images" not in data:
                continue
            images_meta = {img["id"]: img for img in data["images"]}
            # map file names
            file_index = {
                p.name: p
                for p in raw_dir.rglob("*")
                if p.suffix.lower() in IMAGE_EXTS and p.is_file()
            }
            by_image: dict[int, list] = {}
            for ann in data["annotations"]:
                by_image.setdefault(ann["image_id"], []).append(ann)
            for image_id, anns in by_image.items():
                info = images_meta.get(image_id)
                if not info:
                    continue
                path = file_index.get(info["file_name"]) or file_index.get(Path(info["file_name"]).name)
                if path is None:
                    continue
                w = int(info.get("width") or 0)
                h = int(info.get("height") or 0)
                if w <= 0 or h <= 0:
                    try:
                        w, h = image_size(path)
                    except OSError:
                        continue
                boxes = []
                for ann in anns:
                    if "bbox" not in ann:
                        continue
                    x, y, bw, bh = ann["bbox"]
                    boxes.append(xywh_pixels_to_yolo(x, y, bw, bh, w, h, class_id=class_id))
                if boxes:
                    samples.append(
                        Sample(image_path=path, boxes=tuple(boxes), source=self.name)
                    )
        return samples
