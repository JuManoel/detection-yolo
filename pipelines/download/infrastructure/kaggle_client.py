from __future__ import annotations

import logging
from pathlib import Path

from pipelines.shared.infrastructure.fs import ensure_dir, slug_from_kaggle

logger = logging.getLogger(__name__)


class KaggleClient:
    """Download Kaggle datasets into raw_dir/<slug>/."""

    def download(self, dataset: str, raw_dir: Path) -> Path:
        slug = slug_from_kaggle(dataset)
        target = ensure_dir(raw_dir / slug)
        marker = target / ".download_complete"
        if marker.exists() and any(target.iterdir()):
            logger.info("Skipping existing Kaggle dataset: %s", dataset)
            return target

        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError as exc:
            raise RuntimeError(
                "kaggle package is required. Run `uv sync` and configure credentials."
            ) from exc

        api = KaggleApi()
        api.authenticate()
        logger.info("Downloading Kaggle dataset %s → %s", dataset, target)
        api.dataset_download_files(dataset, path=str(target), unzip=True, quiet=False)
        marker.touch()
        return target
