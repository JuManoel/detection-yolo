from __future__ import annotations

import argparse
import logging
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO26n/m/x on unified bird dataset.")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs (default 100)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    from dataclasses import replace

    from pipelines.shared.domain.config import DEFAULT_TRAIN_CONFIG
    from pipelines.train.application.pipeline import TrainPipeline

    config = DEFAULT_TRAIN_CONFIG
    if args.epochs is not None:
        patience = max(1, int(args.epochs * 0.1))
        config = replace(config, epochs=args.epochs, patience=patience)

    try:
        results = TrainPipeline(config=config).run()
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).exception("Train failed: %s", exc)
        return 1

    failed = [r for r in results if not r.success]
    for r in results:
        status = "OK" if r.success else "FAIL"
        print(f"[{status}] {r.model_stem} batch={r.final_batch} weights={r.best_weights}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
