from __future__ import annotations

import json
import logging
from pathlib import Path

from pipelines.metrics.domain.metrics_spec import MetricReport
from pipelines.shared.infrastructure.fs import ensure_dir

logger = logging.getLogger(__name__)


class ReportWriter:
    def write(self, metrics_dir: Path, reports: list[MetricReport]) -> Path:
        ensure_dir(metrics_dir)
        for report in reports:
            path = metrics_dir / f"{report.model_stem}.json"
            path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
            logger.info("Wrote %s", path)

        summary = metrics_dir / "summary.md"
        lines = [
            "# Bird detection metrics",
            "",
            "| Model | Val images | Precision | Recall (bird) | mAP50 | mAP50-95 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in reports:
            lines.append(
                f"| {r.model_stem} | {r.val_images} | {_fmt(r.precision)} | "
                f"{_fmt(r.recall)} | {_fmt(r.map50)} | {_fmt(r.map50_95)} |"
            )
        lines.append("")
        summary.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Wrote %s", summary)
        return summary


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"
