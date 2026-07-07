"""Fase 1 del pipeline: carga y validación de los CSVs.

Responsabilidades:
- Verificar la integridad de `messages.csv.gz` (dataset completo, 24 meses)
  por MD5.
- Cargar los 4 CSVs con dtypes explícitos y conversión de booleanos 't'/'f'.
- Calcular las ventanas temporales (D1 en PROGRESO.md): 18 meses de observación
  y 6 meses de evaluación, ambos contados hacia atrás desde `max(sent_at)`.
- Persistir las fechas calculadas en `config.json::ventana_fechas`.
- Emitir un checkpoint con filas, usuarios únicos y rangos temporales.

DECISIÓN (D5 en PROGRESO.md): se trabaja con `messages.csv.gz` (21.5 GB
comprimido, 24 meses, ~721M filas) en lugar de `messages-demo.csv` (10M filas,
46 días). El archivo se procesa SIEMPRE desde gzip por streaming/chunks; no
se descomprime a disco porque el plano ocuparía ~150 GB y solo hay ~70 GB
libres.

Ejecutar standalone:
    python -m src.data
"""
from __future__ import annotations

import hashlib
import time
from typing import Iterator

import pandas as pd

from src.logging_utils import (
    DIR_DATA,
    cargar_config,
    configurar_logger,
    guardar_config,
    imprimir_checkpoint,
)

# ----------------------------------------------------------------------------
# Esquema de messages.csv.gz
# ----------------------------------------------------------------------------

# DECISIÓN: `client_id` se almacena como int64. Los IDs reales del dataset son
# ~1.5e18, muy por encima del máximo de int32 (~2.1e9). Se conserva la regla
# de tipado explícito de CLAUDE.md, pero se sube de int32 a int64 por el rango
# observado en el dataset real.
DTYPES_MENSAJES: dict[str, str] = {
    "id": "int64",
    "message_id": "string",
    "campaign_id": "Int32",  # nullable: hay registros transaccionales sin campaña
    "message_type": "category",
    "client_id": "int64",
    "channel": "category",
    "category": "category",
    "platform": "category",
    "email_provider": "category",
    "stream": "category",
}

# Las 8 columnas booleanas vienen como 't'/'f' en el CSV. Se convierten a int8
# en post-procesamiento por chunk (más rápido que un converter en read_csv).
COLUMNAS_BOOLEANAS: list[str] = [
    "is_opened",
    "is_clicked",
    "is_unsubscribed",
    "is_hard_bounced",
    "is_soft_bounced",
    "is_complained",
    "is_blocked",
    "is_purchased",
]

# Todas las columnas tipo timestamp del archivo de mensajes.
COLUMNAS_FECHAS_MENSAJES: list[str] = [
    "date",
    "sent_at",
    "opened_first_time_at",
    "opened_last_time_at",
    "clicked_first_time_at",
    "clicked_last_time_at",
    "unsubscribed_at",
    "hard_bounced_at",
    "soft_bounced_at",
    "complained_at",
    "blocked_at",
    "purchased_at",
    "created_at",
    "updated_at",
]

# Subset de columnas usado por Fase 1 para validar integridad y calcular
# ventanas temporales. Se leen solo estas 3 para no cargar 32 columnas × 721M
# filas a memoria.
COLUMNAS_VALIDACION_MINIMA: list[str] = ["client_id", "channel", "sent_at"]

NOMBRE_ARCHIVO_MENSAJES = "messages.csv.gz"

# DECISIÓN: el threshold defensivo se baja a 400M filas (paper reporta ~721M en
# bruto, pero queremos un guardarraíl conservador que solo dispare si la
# descarga quedó truncada). Si la descarga del dataset full quedó parcial,
# Fase 1 corta y avisa.
FILAS_ESPERADAS_MENSAJES = 400_000_000


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _convertir_booleanos_tf(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """Convierte columnas 't'/'f' a int8 (1/0). Trata NaN como 0."""
    for col in columnas:
        if col in df.columns:
            # Comparación vectorizada: cualquier valor distinto de 't' (incluido NaN) -> 0.
            df[col] = (df[col] == "t").astype("int8")
    return df


def _filtrar_dtypes(columnas: list[str] | None) -> dict[str, str]:
    """Devuelve el subset de DTYPES_MENSAJES aplicable a las columnas pedidas."""
    if columnas is None:
        return dict(DTYPES_MENSAJES)
    return {k: v for k, v in DTYPES_MENSAJES.items() if k in columnas}


def _filtrar_parse_dates(columnas: list[str] | None) -> list[str]:
    """Devuelve solo las columnas de fecha que están en `columnas`."""
    if columnas is None:
        return list(COLUMNAS_FECHAS_MENSAJES)
    return [c for c in COLUMNAS_FECHAS_MENSAJES if c in columnas]


def _kwargs_lectura_mensajes(columnas: list[str] | None) -> dict:
    """Arma los kwargs comunes para `pd.read_csv` sobre `messages.csv.gz`."""
    ruta = DIR_DATA / NOMBRE_ARCHIVO_MENSAJES
    dtype_map = _filtrar_dtypes(columnas)
    fechas = _filtrar_parse_dates(columnas)
    # DECISIÓN: pasamos `compression='gzip'` explícito para no depender de la
    # inferencia por extensión (también deja al lector documentado).
    return {
        "filepath_or_buffer": ruta,
        "usecols": columnas,
        "dtype": dtype_map,
        "parse_dates": fechas if fechas else None,
        "compression": "gzip",
    }


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------


def cargar_mensajes(
    columnas: list[str] | None = None,
    chunksize: int | None = None,
) -> pd.DataFrame:
    """Carga `messages.csv.gz` con dtypes/fechas/booleanos resueltos.

    Args:
        columnas: subset de columnas a leer (`usecols`). Para subsets chicos
            (3-4 columnas) esto sigue siendo viable. Para leer las 32 columnas
            completas usar `iterar_mensajes()`.
        chunksize: si está dado, lee en chunks y concatena al final. Pensado
            para subsets que sí caben en RAM una vez concatenados.

    Returns:
        DataFrame con las columnas pedidas, fechas como datetime64 y
        booleanos como int8.

    Raises:
        RuntimeError: si se pide leer todas las columnas en un solo DataFrame.
            El dataset completo descomprimido ronda los 100-150 GB y no entra
            en RAM. Para ese caso usar `iterar_mensajes()`.
    """
    if columnas is None:
        raise RuntimeError(
            "cargar_mensajes() no admite leer todas las columnas a memoria "
            "porque `messages.csv.gz` descomprime a ~100-150 GB. "
            "Usar `iterar_mensajes(columnas=[...])` y agregar por chunks."
        )

    kwargs = _kwargs_lectura_mensajes(columnas)
    bools = [c for c in COLUMNAS_BOOLEANAS if c in columnas]

    if chunksize is None:
        df = pd.read_csv(**kwargs)
        return _convertir_booleanos_tf(df, bools)

    partes: list[pd.DataFrame] = []
    for chunk in pd.read_csv(chunksize=chunksize, **kwargs):
        partes.append(_convertir_booleanos_tf(chunk, bools))
    return pd.concat(partes, ignore_index=True)


def iterar_mensajes(
    columnas: list[str],
    chunksize: int | None = None,
) -> Iterator[pd.DataFrame]:
    """Itera `messages.csv.gz` en chunks, con dtypes/fechas/booleanos resueltos.

    Pensado para Fase 1 (agregación de métricas globales sin acumular el
    DataFrame entero) y Fase 2 (agregación por `client_id` incremental).

    Args:
        columnas: subset de columnas a leer. Requerido para que el consumidor
            controle el ancho del DataFrame por chunk.
        chunksize: filas por chunk. Si es `None`, se toma `chunksize_messages`
            del `config.json`.

    Yields:
        DataFrames de a lo sumo `chunksize` filas, con dtypes ya aplicados y
        booleanos convertidos a int8.
    """
    if not columnas:
        raise ValueError("iterar_mensajes() requiere `columnas` no vacía.")

    if chunksize is None:
        chunksize = int(cargar_config().get("chunksize_messages", 500_000))

    kwargs = _kwargs_lectura_mensajes(columnas)
    bools = [c for c in COLUMNAS_BOOLEANAS if c in columnas]
    for chunk in pd.read_csv(chunksize=chunksize, **kwargs):
        yield _convertir_booleanos_tf(chunk, bools)


def cargar_campaigns() -> pd.DataFrame:
    """Carga `campaigns.csv` (~1.9k filas, sin presión de memoria)."""
    return pd.read_csv(
        DIR_DATA / "campaigns.csv",
        parse_dates=["started_at", "finished_at"],
        dtype={
            "id": "int32",
            "campaign_type": "category",
            "channel": "category",
            "topic": "category",
        },
    )


def cargar_primera_compra() -> pd.DataFrame:
    """Carga `client_first_purchase_date.csv` (~1.85M filas).

    Cubre el rango 2021-12-15 → 2023-12-14, consistente con la ventana del
    dataset full. Se usa para la feature de antigüedad en Fase 2.
    """
    return pd.read_csv(
        DIR_DATA / "client_first_purchase_date.csv",
        parse_dates=["first_purchase_date"],
        dtype={"client_id": "int64"},
    )


def cargar_holidays() -> pd.DataFrame:
    """Carga `holidays.csv` (~47 filas, calendario comercial)."""
    return pd.read_csv(
        DIR_DATA / "holidays.csv",
        parse_dates=["date"],
        dtype={"holiday": "category"},
    )


# ----------------------------------------------------------------------------
# Verificación MD5
# ----------------------------------------------------------------------------


def verificar_md5_mensajes(logger) -> str:
    """Calcula el MD5 de `messages.csv.gz` y lo compara con `config.json`.

    Si el config tiene un MD5 registrado y no coincide, emite warning pero no
    aborta: puede ser una versión del dataset legítimamente distinta.
    """
    config = cargar_config()
    md5_esperado = config.get("dataset_md5", {}).get(NOMBRE_ARCHIVO_MENSAJES)

    ruta = DIR_DATA / NOMBRE_ARCHIVO_MENSAJES
    hasher = hashlib.md5()
    # Lectura en bloques de 8 MB: balance entre throughput y RAM. Sobre 21.5
    # GB esto tarda ~30-60 s en SSD.
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(8 * 1024 * 1024), b""):
            hasher.update(bloque)
    md5_calculado = hasher.hexdigest()

    if md5_esperado is None:
        logger.info("MD5 calculado (no había referencia previa): %s", md5_calculado)
    elif md5_calculado == md5_esperado:
        logger.info("MD5 verificado OK: %s", md5_calculado)
    else:
        logger.warning(
            "MD5 NO coincide. Esperado=%s | Calculado=%s. "
            "El dataset puede haber cambiado; revisar antes de continuar.",
            md5_esperado,
            md5_calculado,
        )
    return md5_calculado


# ----------------------------------------------------------------------------
# Agregación por chunks para Fase 1
# ----------------------------------------------------------------------------


def recolectar_metricas_globales(logger) -> dict:
    """Itera todo `messages.csv.gz` (3 columnas) y agrega métricas globales.

    Esta función reemplaza el patrón anterior de "cargar todo a un DataFrame y
    calcular sobre él". Sobre el dataset full eso explotaría la RAM. Acá se
    acumula: min/max de `sent_at`, set de `client_id` únicos, sumatoria de
    `channel.value_counts()` y conteo total de filas.

    Returns:
        Dict con n_filas, n_usuarios, canales (dict canal -> conteo),
        min_sent_at, max_sent_at.
    """
    n_filas = 0
    n_chunks = 0
    clientes_unicos: set[int] = set()
    canales_acum = pd.Series(dtype="int64")
    min_dt: pd.Timestamp | None = None
    max_dt: pd.Timestamp | None = None

    t_iter = time.time()
    for chunk in iterar_mensajes(columnas=COLUMNAS_VALIDACION_MINIMA):
        n_chunks += 1
        n_filas += len(chunk)
        clientes_unicos.update(chunk["client_id"].unique().tolist())

        conteo_chunk = chunk["channel"].value_counts()
        canales_acum = canales_acum.add(conteo_chunk, fill_value=0)

        chunk_min = chunk["sent_at"].min()
        chunk_max = chunk["sent_at"].max()
        if pd.notna(chunk_min) and (min_dt is None or chunk_min < min_dt):
            min_dt = chunk_min
        if pd.notna(chunk_max) and (max_dt is None or chunk_max > max_dt):
            max_dt = chunk_max

        # Log cada 50 chunks (~25M filas con chunksize=500k) para no spammear.
        if n_chunks % 50 == 0:
            elapsed = time.time() - t_iter
            tasa = n_filas / max(elapsed, 1e-6) / 1_000_000
            logger.info(
                "  ... %d chunks, %d filas acumuladas (%.1f M filas/s)",
                n_chunks,
                n_filas,
                tasa,
            )

    canales_dict = {str(k): int(v) for k, v in canales_acum.astype("int64").items()}
    return {
        "n_filas": n_filas,
        "n_usuarios": len(clientes_unicos),
        "canales": canales_dict,
        "min_sent_at": min_dt,
        "max_sent_at": max_dt,
    }


# ----------------------------------------------------------------------------
# Ventanas temporales (D1)
# ----------------------------------------------------------------------------


def determinar_ventanas_temporales(
    min_dt: pd.Timestamp, max_dt: pd.Timestamp, logger
) -> dict:
    """Calcula `corte_observacion` y `corte_evaluacion` según D1.

    Regla: contar hacia atrás desde `max(sent_at)`. Los últimos 6 meses son
    la ventana de evaluación; los 18 meses anteriores, la ventana de
    observación. Todo lo previo a esos 24 meses se descarta en Fase 2.
    """
    # DECISIÓN: ventanas fijas hacia atrás desde max(sent_at). 6m para evaluación,
    # 18m previos para observación. Se descarta lo anterior. Decisión del usuario,
    # ver D1 en PROGRESO.md. Garantiza ventanas exactas y consistentes con el
    # horizonte de predicción declarado en el paper (sec. IV.B).
    config = cargar_config()
    meses_obs = config["ventana_observacion_meses"]
    meses_eval = config["ventana_evaluacion_meses"]

    corte_eval: pd.Timestamp = max_dt - pd.DateOffset(months=meses_eval)
    corte_obs: pd.Timestamp = corte_eval - pd.DateOffset(months=meses_obs)

    logger.info("Rango temporal del dataset: %s -> %s", min_dt, max_dt)
    logger.info("Corte de observación (inicio de la ventana): %s", corte_obs)
    logger.info("Corte de evaluación  (inicio de la ventana): %s", corte_eval)

    return {
        "min_sent_at": min_dt,
        "max_sent_at": max_dt,
        "corte_observacion": corte_obs,
        "corte_evaluacion": corte_eval,
    }


def persistir_ventanas_en_config(ventanas: dict, md5_actual: str, logger) -> None:
    """Escribe las fechas y el MD5 efectivo en `config.json`.

    Limpia la entrada vieja `messages-demo.csv` del bloque `dataset_md5` y
    deja una sola clave canónica `messages.csv.gz`.
    """
    config = cargar_config()
    config["ventana_fechas"] = {
        "_comentario": (
            "Calculado en Fase 1 desde min/max de sent_at del dataset full "
            "(messages.csv.gz). Ver D1 y D5 en PROGRESO.md."
        ),
        "min_sent_at": ventanas["min_sent_at"].isoformat(),
        "max_sent_at": ventanas["max_sent_at"].isoformat(),
        "corte_observacion": ventanas["corte_observacion"].isoformat(),
        "corte_evaluacion": ventanas["corte_evaluacion"].isoformat(),
    }
    md5_block = config.setdefault("dataset_md5", {})
    md5_block.pop("messages-demo.csv", None)
    md5_block[NOMBRE_ARCHIVO_MENSAJES] = md5_actual
    guardar_config(config)
    logger.info("Fechas y MD5 persistidos en config.json")


# ----------------------------------------------------------------------------
# Validaciones
# ----------------------------------------------------------------------------


def validar_metricas(metricas: dict, logger) -> None:
    """Guardarraíl defensivo sobre las métricas globales del dataset.

    Aborta si el conteo de filas es sospechosamente bajo (regla de CLAUDE.md:
    "Si el dataset cargado tiene menos filas de las esperadas, parar...").
    """
    n_filas = metricas["n_filas"]
    if n_filas < FILAS_ESPERADAS_MENSAJES:
        logger.error(
            "Filas inesperadamente bajas: %d (esperadas >=%d). "
            "La descarga puede estar truncada.",
            n_filas,
            FILAS_ESPERADAS_MENSAJES,
        )
        raise RuntimeError(
            f"messages.csv.gz tiene {n_filas} filas, muy por debajo del "
            f"umbral {FILAS_ESPERADAS_MENSAJES}."
        )


# ----------------------------------------------------------------------------
# Orquestador de Fase 1
# ----------------------------------------------------------------------------


def ejecutar_fase_1() -> None:
    """Punto de entrada de la Fase 1. Validar, calcular ventanas y persistir."""
    logger = configurar_logger("fase_1_data")
    t0 = time.time()
    logger.info("=== INICIO FASE 1 - carga y validacion ===")

    # 1) Verificación MD5 (no fatal si difiere; solo warning).
    md5_actual = verificar_md5_mensajes(logger)

    # 2) Pasada por chunks sobre messages.csv.gz: solo 3 columnas para validar
    #    integridad y calcular el rango temporal. Acumulamos métricas sin
    #    materializar el DataFrame completo (RAM lo prohíbe a esta escala).
    logger.info(
        "Iterando messages.csv.gz por chunks (columnas: %s)...",
        COLUMNAS_VALIDACION_MINIMA,
    )
    t_msg = time.time()
    metricas = recolectar_metricas_globales(logger)
    logger.info(
        "Iteración completa en %.1fs (%d filas)",
        time.time() - t_msg,
        metricas["n_filas"],
    )

    validar_metricas(metricas, logger)

    # 3) Ventanas temporales (D1) + persistencia en config.json.
    ventanas = determinar_ventanas_temporales(
        metricas["min_sent_at"], metricas["max_sent_at"], logger
    )
    persistir_ventanas_en_config(ventanas, md5_actual, logger)

    # 4) Carga + validación rápida de los 3 auxiliares.
    df_campaigns = cargar_campaigns()
    df_primera_compra = cargar_primera_compra()
    df_holidays = cargar_holidays()
    logger.info(
        "Auxiliares cargados: campaigns=%d, primera_compra=%d, holidays=%d",
        len(df_campaigns),
        len(df_primera_compra),
        len(df_holidays),
    )

    # 5) Checkpoint final.
    elapsed = time.time() - t0
    imprimir_checkpoint(
        logger,
        "Fase 1 - Carga y validacion",
        {
            "Filas en messages.csv.gz": f"{metricas['n_filas']:,}",
            "Usuarios únicos (client_id)": f"{metricas['n_usuarios']:,}",
            "Distribución por canal": metricas["canales"],
            "Filas en campaigns": len(df_campaigns),
            "Filas en client_first_purchase_date": f"{len(df_primera_compra):,}",
            "Filas en holidays": len(df_holidays),
            "Rango temporal del dataset": f"{ventanas['min_sent_at']} -> {ventanas['max_sent_at']}",
            "Corte de observación": ventanas["corte_observacion"],
            "Corte de evaluación": ventanas["corte_evaluacion"],
            "MD5 messages.csv.gz": md5_actual,
            "Tiempo total (s)": round(elapsed, 1),
            "Siguiente fase": "Fase 2 - src/features.py",
        },
    )


if __name__ == "__main__":
    ejecutar_fase_1()
