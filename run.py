"""Entrypoint for the communication churn pipeline.

Runs the full pipeline or a single phase with `--phase`. Each phase honors
its own artifact caching; `--force` invalidates it only where applicable
(currently Phase 2: features parquet).

Usage:
    python run.py                          # runs all phases in order
    python run.py --phase data             # Phase 1 only
    python run.py --phase features         # Phase 2 only
    python run.py --phase train            # Phase 3 only
    python run.py --phase evaluate         # Phase 4 only
    python run.py --phase features --force # recompute features ignoring cache

Phases 3 (train) and 4 (evaluate) generate timestamped artifacts on every
run and have no cache of their own; `--force` does not affect them.
"""
from __future__ import annotations

import argparse
import sys
import time

from src.data import run_phase_1
from src.evaluate import run_phase_4
from src.features import run_phase_2
from src.logging_utils import setup_logger
from src.train import run_phase_3


PHASES = ("data", "features", "train", "evaluate", "all")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multichannel CRM communication churn pipeline."
    )
    parser.add_argument(
        "--phase",
        choices=PHASES,
        default="all",
        help="Phase to run. 'all' runs the 4 phases in order (default).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore artifact cache (mainly affects Phase 2).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logger = setup_logger("run")
    t0 = time.time()
    logger.info("== run.py | phase=%s | force=%s ==", args.phase, args.force)

    try:
        if args.phase in ("data", "all"):
            run_phase_1()
        if args.phase in ("features", "all"):
            run_phase_2(force=args.force)
        if args.phase in ("train", "all"):
            run_phase_3(force=args.force)
        if args.phase in ("evaluate", "all"):
            run_phase_4(force=args.force)
    except Exception as exc:
        logger.exception("Pipeline aborted: %s", exc)
        return 1

    logger.info("== pipeline OK in %.1fs ==", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
