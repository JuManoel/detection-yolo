from __future__ import annotations

import logging
from pathlib import Path

from pipelines.shared.infrastructure.env import require_kaggle_api_token
from pipelines.shared.infrastructure.fs import ensure_dir, slug_from_kaggle

logger = logging.getLogger(__name__)


class KaggleClient:
    """Download Kaggle datasets into raw_dir/<slug>/ using KAGGLE_API_TOKEN."""

    def download(self, dataset: str, raw_dir: Path) -> Path:
        slug = slug_from_kaggle(dataset)
        target = ensure_dir(raw_dir / slug)
        marker = target / ".download_complete"
        if marker.exists() and any(p.name != ".download_complete" for p in target.iterdir()):
            logger.info("Skipping existing Kaggle dataset: %s", dataset)
            return target

        # Load .env BEFORE importing kaggle (token is read/consumed on import).
        require_kaggle_api_token()
        import kaggle  # noqa: WPS451 — intentional late import

        api = kaggle.api
        logger.info("Downloading Kaggle dataset %s → %s", dataset, target)
        api.dataset_download_files(dataset, path=str(target), unzip=True, quiet=False)
        marker.touch()
        return target
