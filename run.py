"""Entrypoint del pipeline de churn de comunicación.

Permite ejecutar el pipeline completo o una fase puntual con `--phase`.
Cada fase respeta su propio caching de artefactos; `--force` lo invalida
solo donde aplica (actualmente Fase 2: features parquet).

Uso:
    python run.py                          # corre todas las fases en orden
    python run.py --phase data             # solo Fase 1
    python run.py --phase features         # solo Fase 2
    python run.py --phase train            # solo Fase 3
    python run.py --phase evaluate         # solo Fase 4
    python run.py --phase features --force # recalcula features ignorando cache

Las fases 3 (train) y 4 (evaluate) generan artefactos con timestamp en cada
corrida, no tienen "cache" propio. `--force` no afecta su comportamiento.
"""
from __future__ import annotations

import argparse
import sys
import time

from src.data import ejecutar_fase_1
from src.evaluate import ejecutar_fase_4
from src.features import ejecutar_fase_2
from src.logging_utils import configurar_logger
from src.train import ejecutar_fase_3


FASES = ("data", "features", "train", "evaluate", "all")


def _parsear_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline churn de comunicación CRM multicanal."
    )
    parser.add_argument(
        "--phase",
        choices=FASES,
        default="all",
        help="Fase a ejecutar. 'all' corre las 4 en orden (default).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignora cache de artefactos (afecta principalmente a Fase 2).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parsear_args()
    logger = configurar_logger("run")
    t0 = time.time()
    logger.info("== run.py | phase=%s | force=%s ==", args.phase, args.force)

    try:
        if args.phase in ("data", "all"):
            ejecutar_fase_1()
        if args.phase in ("features", "all"):
            ejecutar_fase_2(force=args.force)
        if args.phase in ("train", "all"):
            ejecutar_fase_3(force=args.force)
        if args.phase in ("evaluate", "all"):
            ejecutar_fase_4(force=args.force)
    except Exception as exc:
        logger.exception("Pipeline abortado: %s", exc)
        return 1

    logger.info("== pipeline OK en %.1fs ==", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
