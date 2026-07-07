"""Logging y utilidades comunes del pipeline.

Centraliza la configuración de logging para que todos los módulos escriban
en el mismo archivo `outputs/logs/run_<timestamp>.log` y en stdout.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# DECISIÓN: timestamp único por corrida para versionar logs y artefactos
# de modelo. Se calcula una sola vez al importar el módulo.
TIMESTAMP_CORRIDA: str = datetime.now().strftime("%Y%m%d_%H%M%S")

# Raíz del proyecto: src/logging_utils.py -> sube un nivel.
RAIZ_PROYECTO: Path = Path(__file__).resolve().parent.parent
DIR_DATA: Path = RAIZ_PROYECTO / "data"
DIR_OUTPUTS: Path = RAIZ_PROYECTO / "outputs"
DIR_FEATURES: Path = DIR_OUTPUTS / "features"
DIR_MODELS: Path = DIR_OUTPUTS / "models"
DIR_FIGURES: Path = DIR_OUTPUTS / "figures"
DIR_LOGS: Path = DIR_OUTPUTS / "logs"
PATH_CONFIG: Path = RAIZ_PROYECTO / "config.json"

_logger_inicializado = False


def configurar_logger(nombre: str = "pipeline") -> logging.Logger:
    """Devuelve un logger que escribe a archivo y a stdout.

    El archivo de log queda en `outputs/logs/run_<timestamp>.log` y se reutiliza
    durante toda la corrida (el timestamp se fija al importar el módulo).

    DECISIÓN: los handlers se adjuntan al logger ROOT (no al nombrado) y los
    loggers nombrados heredan vía `propagate=True`. Así da igual cuántas veces
    `configurar_logger("foo")`, `configurar_logger("bar")` se llamen desde
    distintos módulos: todos escriben al mismo archivo y stdout sin duplicar
    handlers ni perder mensajes en loggers "hijos".
    """
    global _logger_inicializado
    logger = logging.getLogger(nombre)

    if not _logger_inicializado:
        DIR_LOGS.mkdir(parents=True, exist_ok=True)
        archivo_log = DIR_LOGS / f"run_{TIMESTAMP_CORRIDA}.log"

        formato = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        handler_archivo = logging.FileHandler(archivo_log, encoding="utf-8")
        handler_archivo.setFormatter(formato)
        handler_stdout = logging.StreamHandler(sys.stdout)
        handler_stdout.setFormatter(formato)

        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.addHandler(handler_archivo)
        root.addHandler(handler_stdout)

        _logger_inicializado = True
        logger.info("Logger inicializado. Archivo de log: %s", archivo_log)

    # Aseguramos nivel y propagación del logger nombrado en cada llamada.
    logger.setLevel(logging.INFO)
    logger.propagate = True
    return logger


def cargar_config() -> dict:
    """Lee `config.json` y lo devuelve como dict."""
    with PATH_CONFIG.open("r", encoding="utf-8") as f:
        return json.load(f)


def guardar_config(config: dict) -> None:
    """Sobrescribe `config.json` con la versión actualizada."""
    with PATH_CONFIG.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")


def imprimir_checkpoint(logger: logging.Logger, titulo: str, items: dict) -> None:
    """Imprime un checkpoint estandarizado al cerrar cada fase."""
    barra = "=" * 70
    logger.info(barra)
    logger.info("CHECKPOINT — %s", titulo)
    logger.info(barra)
    for clave, valor in items.items():
        logger.info("  %s: %s", clave, valor)
    logger.info(barra)
