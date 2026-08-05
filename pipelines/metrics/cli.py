from __future__ import annotations

import argparse
import logging
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute bird detection metrics for trained models.")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    from pipelines.metrics.application.pipeline import MetricsPipeline

    try:
        reports = MetricsPipeline().run()
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).exception("Metrics failed: %s", exc)
        return 1

    for r in reports:
        print(
            f"{r.model_stem}: val={r.val_images} P={r.precision} R={r.recall} "
            f"mAP50={r.map50} mAP50-95={r.map50_95}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
