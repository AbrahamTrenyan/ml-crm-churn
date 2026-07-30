"""Logging and shared pipeline utilities.

Centralizes logging configuration so that every module writes to the same
`outputs/logs/run_<timestamp>.log` file and to stdout.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# DECISION: a single timestamp per run to version logs and model artifacts.
# Computed once at module import time.
RUN_TIMESTAMP: str = datetime.now().strftime("%Y%m%d_%H%M%S")

# Project root: src/logging_utils.py -> one level up.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
FEATURES_DIR: Path = OUTPUTS_DIR / "features"
MODELS_DIR: Path = OUTPUTS_DIR / "models"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
LOGS_DIR: Path = OUTPUTS_DIR / "logs"
CONFIG_PATH: Path = PROJECT_ROOT / "config.json"

_logger_initialized = False


def setup_logger(name: str = "pipeline") -> logging.Logger:
    """Returns a logger that writes to a file and to stdout.

    The log file lives at `outputs/logs/run_<timestamp>.log` and is reused
    for the whole run (the timestamp is fixed at module import time).

    DECISION: handlers are attached to the ROOT logger (not the named one)
    and named loggers inherit via `propagate=True`. This way it does not
    matter how many times `setup_logger("foo")`, `setup_logger("bar")` are
    called from different modules: they all write to the same file and
    stdout without duplicating handlers or losing messages in child loggers.
    """
    global _logger_initialized
    logger = logging.getLogger(name)

    if not _logger_initialized:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOGS_DIR / f"run_{RUN_TIMESTAMP}.log"

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)

        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.addHandler(file_handler)
        root.addHandler(stdout_handler)

        _logger_initialized = True
        logger.info("Logger initialized. Log file: %s", log_file)

    # Ensure level and propagation of the named logger on every call.
    logger.setLevel(logging.INFO)
    logger.propagate = True
    return logger


def load_config() -> dict:
    """Reads `config.json` and returns it as a dict."""
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config: dict) -> None:
    """Overwrites `config.json` with the updated version."""
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")


def print_checkpoint(logger: logging.Logger, title: str, items: dict) -> None:
    """Prints a standardized checkpoint when closing each phase."""
    bar = "=" * 70
    logger.info(bar)
    logger.info("CHECKPOINT — %s", title)
    logger.info(bar)
    for key, value in items.items():
        logger.info("  %s: %s", key, value)
    logger.info(bar)
