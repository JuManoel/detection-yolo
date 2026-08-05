from __future__ import annotations

import logging
from pathlib import Path

from pipelines.metrics.domain.metrics_spec import MetricReport

logger = logging.getLogger(__name__)


class YoloValidator:
    """Run Ultralytics validation and extract bird detection metrics."""

    def validate(
        self,
        weights: Path,
        data_yaml: Path,
        *,
        model_stem: str,
        val_images: int,
        device: str | int = "cpu",
    ) -> MetricReport:
        from ultralytics import YOLO

        logger.info("Validating %s on %s", weights, data_yaml)
        model = YOLO(str(weights))
        results = model.val(data=str(data_yaml), device=device, split="val")

        precision = _safe_float(getattr(results.box, "mp", None))
        recall = _safe_float(getattr(results.box, "mr", None))
        map50 = _safe_float(getattr(results.box, "map50", None))
        map50_95 = _safe_float(getattr(results.box, "map", None))

        box_loss = None
        extra: dict = {}
        try:
            # speed / fitness extras when available
            if hasattr(results, "speed"):
                extra["speed"] = dict(results.speed)
            if hasattr(results.box, "maps"):
                extra["per_class_map"] = [float(x) for x in list(results.box.maps)]
        except Exception:  # noqa: BLE001
            pass

        return MetricReport(
            model_stem=model_stem,
            weights=str(weights),
            val_images=val_images,
            precision=precision,
            recall=recall,
            map50=map50,
            map50_95=map50_95,
            box_loss=box_loss,
            extra=extra or None,
        )


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
