from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from pipelines.download.domain.models import Sample
from pipelines.shared.domain.bbox import voc_xyxy_to_yolo
from pipelines.shared.domain.config import DEFAULT_TRAIN_CONFIG

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _index_images(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.suffix.lower() in IMAGE_EXTS and path.is_file():
            index[path.stem] = path
            index[path.name] = path
    return index


class FbdConverter:
    name = "fbd-sv-2024"

    def convert(self, raw_dir: Path) -> list[Sample]:
        xml_files = list(raw_dir.rglob("*.xml"))
        if not xml_files:
            logger.warning("No VOC XML labels found under %s", raw_dir)
            return []

        images = _index_images(raw_dir)
        samples: list[Sample] = []
        class_id = DEFAULT_TRAIN_CONFIG.class_id

        for xml_path in xml_files:
            try:
                root = ET.parse(xml_path).getroot()
            except ET.ParseError:
                continue

            filename = (root.findtext("filename") or xml_path.stem).strip()
            stem = Path(filename).stem
            img_path = images.get(stem) or images.get(filename) or images.get(xml_path.stem)
            if img_path is None:
                # try sibling images folder
                for candidate in (
                    xml_path.with_suffix(".jpg"),
                    xml_path.with_suffix(".jpeg"),
                    xml_path.with_suffix(".png"),
                    xml_path.parent.parent / "images" / f"{stem}.jpg",
                ):
                    if candidate.exists():
                        img_path = candidate
                        break
            if img_path is None or not img_path.exists():
                continue

            size = root.find("size")
            if size is not None:
                img_w = int(float(size.findtext("width") or 0))
                img_h = int(float(size.findtext("height") or 0))
            else:
                from pipelines.download.infrastructure.yolo_writer import image_size

                img_w, img_h = image_size(img_path)

            if img_w <= 0 or img_h <= 0:
                continue

            boxes = []
            for obj in root.findall("object"):
                name = (obj.findtext("name") or "").strip().lower()
                if name and name not in {"bird", "flying_bird", "n01503061"}:
                    # keep unknown object names if dataset is bird-only
                    if "bird" not in name:
                        continue
                bnd = obj.find("bndbox")
                if bnd is None:
                    continue
                try:
                    xmin = float(bnd.findtext("xmin") or 0)
                    ymin = float(bnd.findtext("ymin") or 0)
                    xmax = float(bnd.findtext("xmax") or 0)
                    ymax = float(bnd.findtext("ymax") or 0)
                except ValueError:
                    continue
                boxes.append(
                    voc_xyxy_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h, class_id=class_id)
                )
            if boxes:
                samples.append(Sample(image_path=img_path, boxes=tuple(boxes), source=self.name))

        logger.info("FBD converter: %d samples from %s", len(samples), raw_dir)
        return samples
