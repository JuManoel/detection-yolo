from __future__ import annotations

import argparse
import logging
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Kaggle + COCO bird datasets and build unified YOLO data."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    # Load .env before any kaggle import (KAGGLE_API_TOKEN auth).
    from pipelines.shared.infrastructure.env import load_project_env

    load_project_env()

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    from pipelines.download.application.pipeline import DownloadPipeline

    try:
        yaml_path = DownloadPipeline().run()
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).exception("Download failed: %s", exc)
        return 1
    print(f"YOLO dataset ready: {yaml_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
