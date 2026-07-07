"""Fase 2 del pipeline: feature engineering por usuario.

Construye el vector de features comportamentales descrito en la sección IV.C
del paper (líneas 295-297), iterando `messages.csv.gz` por chunks y agregando
métricas por `client_id` sin materializar el dataset completo en RAM.

Esquema general:
1. Cargar config -> obtener `corte_observacion` y `corte_evaluacion`.
2. Pre-cargar campaigns (lookup por campaign_id) y client_first_purchase_date.
3. Iterar messages.csv.gz por chunks; por cada chunk:
   a. Filtrar a la unión de ventanas (obs + eval).
   b. Marcar es_obs / es_eval con la fecha de corte.
   c. Joinear info de campaña (es_trigger, tiene_personalizacion).
   d. Agregar counts y maxes por (client_id, channel) y por client_id.
   e. Mezclar con acumuladores globales.
4. Calcular features derivadas (tasas, ratios, recencias).
5. Joinear antigüedad del cliente.
6. Construir target binario sobre la ventana de evaluación.
7. Filtrar a usuarios "evaluables" (recibieron al menos un mensaje en eval).
8. Persistir parquet en `outputs/features/features_por_usuario.parquet`.

DECISIÓN: agregación incremental por chunk (no concat global). Cada chunk
produce un DataFrame chico (~max 460k filas únicas por client_id) que se
combina con el acumulador via `.add(fill_value=0)` para sumas y un merge
seguido de `max` para timestamps. Esto evita explotar RAM con 721M filas.

DECISIÓN: el split temporal "fuerte" del pipeline es features (ventana de
observación) vs target (ventana de evaluación). El train/test split dentro
de Fase 3 es estratificado por target -consistente con el paper, sección
IV.D-, no temporal. La regla anti-leakage se cumple porque ninguna feature
mira datos posteriores a `corte_evaluacion`.

Ejecutar standalone:
    python -m src.features
"""
from __future__ import annotations

import gc
import time
from typing import Iterator

import numpy as np
import pandas as pd

from src.data import iterar_mensajes, cargar_campaigns, cargar_primera_compra
from src.logging_utils import (
    DIR_FEATURES,
    cargar_config,
    configurar_logger,
    imprimir_checkpoint,
)

# ----------------------------------------------------------------------------
# Constantes de columnas y canales
# ----------------------------------------------------------------------------

# DECISIÓN: solo se usan los canales con volumen significativo en el dataset
# (paper sec. IV.A línea 287). Resto de canales se descarta del modelado.
CANALES_OBJETIVO: tuple[str, ...] = ("email", "mobile_push")

# Columnas leídas de messages.csv.gz para Fase 2. Conservadoras: las mínimas
# necesarias para construir todas las features de la lista cerrada D3.
COLUMNAS_MENSAJES_FASE2: list[str] = [
    "client_id",
    "campaign_id",
    "message_type",
    "channel",
    "sent_at",
    "is_opened",
    "opened_first_time_at",
    "is_clicked",
    "clicked_first_time_at",
    "is_hard_bounced",
    "is_soft_bounced",
    "is_unsubscribed",
    "is_complained",
    "is_purchased",
]

# Ventanas cortas usadas para "tasa de apertura reciente" (paper sec. IV.C).
DIAS_RECIENTE_CORTOS: int = 30
DIAS_RECIENTE_LARGOS: int = 60


# ----------------------------------------------------------------------------
# Setup: cortes temporales y lookups auxiliares
# ----------------------------------------------------------------------------


def _cargar_cortes_temporales() -> dict:
    """Lee `ventana_fechas` del config y devuelve los Timestamps."""
    config = cargar_config()
    vf = config["ventana_fechas"]
    if vf.get("corte_observacion") is None or vf.get("corte_evaluacion") is None:
        raise RuntimeError(
            "config.json::ventana_fechas no tiene cortes válidos. "
            "Ejecutar primero Fase 1 (python -m src.data)."
        )
    corte_obs = pd.Timestamp(vf["corte_observacion"])
    corte_eval = pd.Timestamp(vf["corte_evaluacion"])
    max_dt = pd.Timestamp(vf["max_sent_at"])
    return {
        "corte_observacion": corte_obs,
        "corte_evaluacion": corte_eval,
        "max_sent_at": max_dt,
        "corte_reciente_30d": corte_eval - pd.Timedelta(days=DIAS_RECIENTE_CORTOS),
        "corte_reciente_60d": corte_eval - pd.Timedelta(days=DIAS_RECIENTE_LARGOS),
    }


def _construir_lookup_campaigns() -> pd.DataFrame:
    """Lookup chico campaign_id -> (es_trigger, tiene_personalizacion).

    Se reduce a un DataFrame indexado por `campaign_id` con dos columnas
    booleanas, para hacer `merge` vectorizado contra cada chunk de mensajes.
    """
    df = cargar_campaigns()
    # DECISIÓN: `campaign_type` es categorical -> comparamos contra str y
    # llevamos a int8. `subject_with_personalization` se lee como object con
    # bool python (True/False) y NaN para campañas no-bulk. Usamos
    # `fillna(False)` + astype("bool") para neutralizar los NaN y casteamos
    # a int8 sin perder semántica.
    es_trigger = (df["campaign_type"].astype("object") == "trigger").astype("int8")
    tiene_personalizacion = (
        df["subject_with_personalization"].fillna(False).astype(bool).astype("int8")
    )
    lookup = pd.DataFrame(
        {
            "es_trigger": es_trigger.to_numpy(),
            "tiene_personalizacion": tiene_personalizacion.to_numpy(),
        },
        index=df["id"].astype("int64"),
    )
    lookup.index.name = "campaign_id"
    return lookup


# ----------------------------------------------------------------------------
# Helpers de agregación
# ----------------------------------------------------------------------------


# DECISIÓN: las sumas se almacenan en `float32` (no float64). Cada celda es
# un conteo de mensajes/eventos con cota práctica de unos pocos miles, muy
# por dentro del rango float32. Esto reduce a la mitad la memoria del
# acumulador (16M usuarios × 29 columnas pasan de ~3.7 GB a ~1.9 GB) y
# evita el swap-thrashing que mató la primera corrida sobre el dataset full.
DTYPE_SUMS: str = "float32"


def _df_vacio_sums() -> pd.DataFrame:
    """Esqueleto de DataFrame de sumas, indexado por client_id."""
    columnas = _columnas_sums()
    return pd.DataFrame(columns=columnas, dtype=DTYPE_SUMS)


def _df_vacio_maxes() -> pd.DataFrame:
    """Esqueleto de DataFrame de máximos de timestamps, indexado por client_id."""
    columnas = _columnas_maxes()
    return pd.DataFrame(columns=columnas, dtype="datetime64[ns]")


def _columnas_sums() -> list[str]:
    """Lista de columnas escalares que se agregan por suma a lo largo de chunks."""
    cols = []
    for canal in CANALES_OBJETIVO:
        cols += [
            f"n_msgs_{canal}",
            f"n_opens_{canal}",
            f"n_clicks_{canal}",
            f"n_hard_bounces_{canal}",
            f"n_soft_bounces_{canal}",
            f"n_unsubs_{canal}",
            f"n_complaints_{canal}",
            f"n_msgs_30d_{canal}",
            f"n_opens_30d_{canal}",
            f"n_msgs_60d_{canal}",
            f"n_opens_60d_{canal}",
        ]
    cols += [
        "n_compras_atribuidas",
        "n_msgs_obs_total",
        "n_msgs_trigger_obs",
        "n_msgs_personalizados_obs",
        "n_msgs_eval",
        "n_opens_eval",
        "n_clicks_eval",
    ]
    return cols


def _columnas_maxes() -> list[str]:
    """Lista de columnas timestamp que se agregan por max a lo largo de chunks."""
    cols = []
    for canal in CANALES_OBJETIVO:
        cols += [f"ultimo_open_{canal}", f"ultimo_click_{canal}"]
    return cols


# ----------------------------------------------------------------------------
# Procesamiento por chunk
# ----------------------------------------------------------------------------


def _enriquecer_chunk(
    chunk: pd.DataFrame,
    cortes: dict,
    lookup_campaigns: pd.DataFrame,
) -> pd.DataFrame:
    """Filtra el chunk a la unión de ventanas y agrega columnas de soporte.

    Agrega:
    - `es_obs`, `es_eval` (mascaras temporales).
    - `es_trigger`, `tiene_personalizacion` (merge con campaigns).
    - `en_30d`, `en_60d` (mascaras de recencia dentro de obs).
    """
    # 1) Filtro temporal: descartar todo lo previo a corte_obs (D1).
    chunk = chunk.loc[chunk["sent_at"] >= cortes["corte_observacion"]]
    if chunk.empty:
        return chunk

    chunk = chunk.assign(
        es_obs=chunk["sent_at"] < cortes["corte_evaluacion"],
        es_eval=chunk["sent_at"] >= cortes["corte_evaluacion"],
    )
    # Mascaras de recencia (solo aplican dentro de obs).
    chunk["en_30d"] = chunk["es_obs"] & (chunk["sent_at"] >= cortes["corte_reciente_30d"])
    chunk["en_60d"] = chunk["es_obs"] & (chunk["sent_at"] >= cortes["corte_reciente_60d"])

    # 2) Merge vectorizado con campaigns (left join; sin match -> 0).
    chunk = chunk.merge(
        lookup_campaigns, how="left", left_on="campaign_id", right_index=True
    )
    chunk["es_trigger"] = chunk["es_trigger"].fillna(0).astype("int8")
    chunk["tiene_personalizacion"] = (
        chunk["tiene_personalizacion"].fillna(0).astype("int8")
    )
    return chunk


def _agregar_sums_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Suma todas las features de tipo 'count' por client_id para el chunk.

    Devuelve un DataFrame indexado por client_id con las columnas de `_columnas_sums`.
    """
    df_obs = chunk.loc[chunk["es_obs"]]
    df_eval = chunk.loc[chunk["es_eval"]]
    out_pieces: list[pd.DataFrame] = []

    # Sumas por canal (solo para canales objetivo).
    for canal in CANALES_OBJETIVO:
        sub = df_obs.loc[df_obs["channel"] == canal]
        if sub.empty:
            continue
        g = sub.groupby("client_id", observed=True)
        piece = pd.DataFrame(
            {
                f"n_msgs_{canal}": g.size(),
                f"n_opens_{canal}": g["is_opened"].sum(),
                f"n_clicks_{canal}": g["is_clicked"].sum(),
                f"n_hard_bounces_{canal}": g["is_hard_bounced"].sum(),
                f"n_soft_bounces_{canal}": g["is_soft_bounced"].sum(),
                f"n_unsubs_{canal}": g["is_unsubscribed"].sum(),
                f"n_complaints_{canal}": g["is_complained"].sum(),
                f"n_msgs_30d_{canal}": sub.loc[sub["en_30d"]].groupby("client_id", observed=True).size(),
                f"n_opens_30d_{canal}": sub.loc[sub["en_30d"]].groupby("client_id", observed=True)["is_opened"].sum(),
                f"n_msgs_60d_{canal}": sub.loc[sub["en_60d"]].groupby("client_id", observed=True).size(),
                f"n_opens_60d_{canal}": sub.loc[sub["en_60d"]].groupby("client_id", observed=True)["is_opened"].sum(),
            }
        )
        out_pieces.append(piece)

    # Sumas globales sobre OBS (no segmentadas por canal).
    if not df_obs.empty:
        g_obs = df_obs.groupby("client_id", observed=True)
        out_pieces.append(
            pd.DataFrame(
                {
                    "n_compras_atribuidas": g_obs["is_purchased"].sum(),
                    "n_msgs_obs_total": g_obs.size(),
                    "n_msgs_trigger_obs": g_obs["es_trigger"].sum(),
                    "n_msgs_personalizados_obs": g_obs["tiene_personalizacion"].sum(),
                }
            )
        )

    # Sumas sobre EVAL (target y soporte).
    if not df_eval.empty:
        g_eval = df_eval.groupby("client_id", observed=True)
        out_pieces.append(
            pd.DataFrame(
                {
                    "n_msgs_eval": g_eval.size(),
                    "n_opens_eval": g_eval["is_opened"].sum(),
                    "n_clicks_eval": g_eval["is_clicked"].sum(),
                }
            )
        )

    if not out_pieces:
        return pd.DataFrame()

    # Unimos por client_id (outer): cada usuario aparece una sola vez.
    unido = pd.concat(out_pieces, axis=1).fillna(0)
    # Casteamos a float32 para reducir 50% la presión de memoria del acumulador.
    return unido.astype(DTYPE_SUMS)


def _agregar_maxes_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Calcula maxes de timestamps de open/click por (client_id, canal) en obs."""
    df_obs = chunk.loc[chunk["es_obs"]]
    cols: dict[str, pd.Series] = {}
    for canal in CANALES_OBJETIVO:
        sub = df_obs.loc[df_obs["channel"] == canal]
        if sub.empty:
            continue
        g = sub.groupby("client_id", observed=True)
        # DECISIÓN: usamos `opened_first_time_at` (NaT si no abrió) y tomamos
        # el max sobre todos los mensajes del usuario en obs. Equivale al
        # último open del usuario en ese canal dentro de obs.
        cols[f"ultimo_open_{canal}"] = g["opened_first_time_at"].max()
        cols[f"ultimo_click_{canal}"] = g["clicked_first_time_at"].max()
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols)


# DECISIÓN: el `acum` (DataFrame indexed por client_id) crece a ~16M filas.
# Combinarlo con cada chunk via `.add(fill_value=0)` o `pd.concat+groupby.max`
# es O(tamaño_acum) por chunk → cuadrático total. Refactor: acumulamos los
# DataFrames "chunk-level" en una lista (`pending`) y solo los compactamos
# cada `CHUNKS_POR_COMPACTACION` chunks. Esto convierte 1442 merges grandes en
# ~70 merges grandes, manteniendo el resultado matemáticamente idéntico.
#
# Valor: 20 (bajamos de 50 tras la primera corrida real, que mostró que
# con 50 chunks pendientes el pico de RAM durante la compactación cruza el
# umbral de swap en una máquina de 16 GB).
CHUNKS_POR_COMPACTACION: int = 20


def _compactar_sums(sums_acum: pd.DataFrame, pendientes: list[pd.DataFrame]) -> pd.DataFrame:
    """Combina el acumulador de sumas con N DataFrames pendientes en un solo groupby."""
    no_vacios = [df for df in pendientes if not df.empty]
    if not no_vacios:
        return sums_acum
    combinado = pd.concat([sums_acum] + no_vacios)
    if combinado.empty:
        return sums_acum
    # `groupby(level=0).sum()` preserva el dtype float32 de la entrada.
    return combinado.groupby(level=0).sum()


def _compactar_maxes(maxes_acum: pd.DataFrame, pendientes: list[pd.DataFrame]) -> pd.DataFrame:
    """Combina el acumulador de timestamps con N DataFrames pendientes en un solo groupby."""
    no_vacios = [df for df in pendientes if not df.empty]
    if not no_vacios:
        return maxes_acum
    combinado = pd.concat([maxes_acum] + no_vacios)
    if combinado.empty:
        return maxes_acum
    return combinado.groupby(level=0).max()


# ----------------------------------------------------------------------------
# Orquestación del scan completo
# ----------------------------------------------------------------------------


def construir_acumuladores(logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scan completo de messages.csv.gz, devolviendo (sums, maxes) por usuario.

    Usa compactación batch (cada `CHUNKS_POR_COMPACTACION` chunks) para
    amortizar el costo de los merges grandes y mantener tiempo de iteración
    aproximadamente lineal en lugar de cuadrático.
    """
    cortes = _cargar_cortes_temporales()
    lookup = _construir_lookup_campaigns()
    logger.info(
        "Cortes temporales: obs=%s, eval=%s, max=%s",
        cortes["corte_observacion"],
        cortes["corte_evaluacion"],
        cortes["max_sent_at"],
    )
    logger.info(
        "Compactacion batch cada %d chunks (acumuladores se mergean por lote).",
        CHUNKS_POR_COMPACTACION,
    )

    sums_acum = _df_vacio_sums()
    maxes_acum = _df_vacio_maxes()
    pend_sums: list[pd.DataFrame] = []
    pend_maxes: list[pd.DataFrame] = []
    n_chunks = 0
    n_filas_proc = 0
    t0 = time.time()

    for chunk in iterar_mensajes(columnas=COLUMNAS_MENSAJES_FASE2):
        n_chunks += 1
        chunk = _enriquecer_chunk(chunk, cortes, lookup)
        if chunk.empty:
            if n_chunks % CHUNKS_POR_COMPACTACION == 0:
                sums_acum = _compactar_sums(sums_acum, pend_sums)
                maxes_acum = _compactar_maxes(maxes_acum, pend_maxes)
                pend_sums, pend_maxes = [], []
            continue
        n_filas_proc += len(chunk)

        pend_sums.append(_agregar_sums_chunk(chunk))
        pend_maxes.append(_agregar_maxes_chunk(chunk))

        if n_chunks % CHUNKS_POR_COMPACTACION == 0:
            t_compact = time.time()
            sums_acum = _compactar_sums(sums_acum, pend_sums)
            maxes_acum = _compactar_maxes(maxes_acum, pend_maxes)
            # Liberamos referencias antes del gc para que efectivamente
            # se reclamen los buffers de los chunks pendientes.
            pend_sums.clear()
            pend_maxes.clear()
            gc.collect()
            elapsed = time.time() - t0
            logger.info(
                "  ... %d chunks, %d usuarios acumulados, %.1fs (compactacion %.1fs)",
                n_chunks,
                len(sums_acum),
                elapsed,
                time.time() - t_compact,
            )

    # Compactacion final por si quedaron pendientes.
    if pend_sums or pend_maxes:
        sums_acum = _compactar_sums(sums_acum, pend_sums)
        maxes_acum = _compactar_maxes(maxes_acum, pend_maxes)

    logger.info(
        "Scan terminado: %d chunks, %d filas dentro de ventanas, %d usuarios.",
        n_chunks,
        n_filas_proc,
        len(sums_acum),
    )
    return sums_acum, maxes_acum


# ----------------------------------------------------------------------------
# Features derivadas
# ----------------------------------------------------------------------------


def _calcular_tasas_y_recencias(
    df: pd.DataFrame, cortes: dict
) -> pd.DataFrame:
    """Agrega tasas (open/click), tasas recientes y días-desde-última-acción."""
    corte_eval = cortes["corte_evaluacion"]

    for canal in CANALES_OBJETIVO:
        # Tasas (NaN si no recibió mensajes en el canal o ventana).
        df[f"tasa_open_{canal}"] = df[f"n_opens_{canal}"] / df[f"n_msgs_{canal}"].replace(0, np.nan)
        df[f"tasa_click_{canal}"] = df[f"n_clicks_{canal}"] / df[f"n_msgs_{canal}"].replace(0, np.nan)
        df[f"tasa_open_30d_{canal}"] = (
            df[f"n_opens_30d_{canal}"] / df[f"n_msgs_30d_{canal}"].replace(0, np.nan)
        )
        df[f"tasa_open_60d_{canal}"] = (
            df[f"n_opens_60d_{canal}"] / df[f"n_msgs_60d_{canal}"].replace(0, np.nan)
        )
        # Recencias: días desde el último open/click hasta el cierre de obs.
        # NaN si el usuario nunca abrió/clickeó en ese canal.
        df[f"dias_desde_ultimo_open_{canal}"] = (
            (corte_eval - df[f"ultimo_open_{canal}"]).dt.total_seconds() / 86400.0
        )
        df[f"dias_desde_ultimo_click_{canal}"] = (
            (corte_eval - df[f"ultimo_click_{canal}"]).dt.total_seconds() / 86400.0
        )
    return df


def _calcular_features_cross_channel(df: pd.DataFrame, cortes: dict) -> pd.DataFrame:
    """Features cross-channel: nº de canales con open, diff de tasas, ratio reciente."""
    # n_canales_con_open: cuántos canales del conjunto objetivo tienen >=1 open.
    flags = pd.concat(
        [(df[f"n_opens_{c}"] > 0).astype("int8") for c in CANALES_OBJETIVO], axis=1
    )
    df["n_canales_con_open"] = flags.sum(axis=1).astype("int8")

    # diff_tasa_open: diferencia entre tasas por canal (paper). Se reporta
    # email - mobile_push; NaN si alguno de los dos canales no tiene mensajes.
    df["diff_tasa_open_email_vs_push"] = (
        df["tasa_open_email"] - df["tasa_open_mobile_push"]
    )

    # ratio_actividad_30d_vs_historico: opens últimos 30 días vs promedio
    # histórico esperado bajo distribución uniforme dentro de obs.
    dias_obs = (
        cortes["corte_evaluacion"] - cortes["corte_observacion"]
    ).total_seconds() / 86400.0
    opens_30d_total = sum(df[f"n_opens_30d_{c}"] for c in CANALES_OBJETIVO)
    opens_obs_total = sum(df[f"n_opens_{c}"] for c in CANALES_OBJETIVO)
    esperado_30d = opens_obs_total * (DIAS_RECIENTE_CORTOS / dias_obs)
    df["ratio_actividad_30d_vs_historico"] = opens_30d_total / esperado_30d.replace(
        0, np.nan
    )
    return df


def _calcular_features_campania(df: pd.DataFrame) -> pd.DataFrame:
    """Proporciones de mensajes trigger y con personalización (paper sec. IV.C)."""
    total = df["n_msgs_obs_total"].replace(0, np.nan)
    df["prop_trigger"] = df["n_msgs_trigger_obs"] / total
    df["prop_personalizacion"] = df["n_msgs_personalizados_obs"] / total
    return df


def _agregar_antiguedad(df: pd.DataFrame, cortes: dict) -> pd.DataFrame:
    """Joinea `client_first_purchase_date` y calcula días de antigüedad."""
    primera = cargar_primera_compra().set_index("client_id")
    df = df.merge(
        primera, how="left", left_index=True, right_index=True
    )
    df["dias_antiguedad"] = (
        (cortes["corte_observacion"] - df["first_purchase_date"]).dt.total_seconds()
        / 86400.0
    )
    df = df.drop(columns=["first_purchase_date"])
    return df


def _construir_target(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Define target binario y filtra a usuarios evaluables.

    target_churn_comunicacion = 1 si el usuario recibió >=1 mensaje en la
    ventana de evaluación pero no abrió ni clickeó ninguno; 0 en caso
    contrario. Usuarios sin mensajes en eval se descartan (no son evaluables).
    """
    n_antes = len(df)
    df["n_msgs_eval"] = df["n_msgs_eval"].fillna(0)
    df["n_opens_eval"] = df["n_opens_eval"].fillna(0)
    df["n_clicks_eval"] = df["n_clicks_eval"].fillna(0)

    evaluables = df["n_msgs_eval"] > 0
    df = df.loc[evaluables].copy()
    n_filtrados = n_antes - len(df)
    logger.info(
        "Filtrados %d usuarios sin mensajes en eval. Evaluables: %d",
        n_filtrados,
        len(df),
    )

    df["target_churn_comunicacion"] = (
        (df["n_opens_eval"] == 0) & (df["n_clicks_eval"] == 0)
    ).astype("int8")
    return df


# ----------------------------------------------------------------------------
# Persistencia
# ----------------------------------------------------------------------------


PATH_FEATURES = DIR_FEATURES / "features_por_usuario.parquet"


def _persistir(df: pd.DataFrame, logger) -> None:
    """Escribe el DataFrame final a parquet (pyarrow)."""
    DIR_FEATURES.mkdir(parents=True, exist_ok=True)
    df = df.reset_index().rename(columns={"index": "client_id"})
    df["client_id"] = df["client_id"].astype("int64")
    df.to_parquet(PATH_FEATURES, engine="pyarrow", index=False)
    logger.info("Features persistidas en %s", PATH_FEATURES)


# ----------------------------------------------------------------------------
# Orquestador de Fase 2
# ----------------------------------------------------------------------------


def ejecutar_fase_2(force: bool = False) -> None:
    """Punto de entrada de la Fase 2. Construye y persiste features por usuario.

    Si el parquet ya existe y no se pasa `force=True`, no recalcula.
    """
    logger = configurar_logger("fase_2_features")
    t0 = time.time()
    logger.info("=== INICIO FASE 2 - feature engineering ===")

    if PATH_FEATURES.exists() and not force:
        logger.info("Parquet ya existe (%s). Usar force=True para recalcular.", PATH_FEATURES)
        return

    cortes = _cargar_cortes_temporales()

    # 1) Scan + agregación.
    sums_acum, maxes_acum = construir_acumuladores(logger)

    # 2) Unir sums + maxes en un solo DataFrame por usuario.
    df = sums_acum.join(maxes_acum, how="outer")
    df.index.name = "client_id"
    # Las columnas de sum que quedaron NaN tras el outer join son ceros reales.
    for col in _columnas_sums():
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 3) Features derivadas y joins finales.
    df = _calcular_tasas_y_recencias(df, cortes)
    df = _calcular_features_cross_channel(df, cortes)
    df = _calcular_features_campania(df)
    df = _agregar_antiguedad(df, cortes)
    df = _construir_target(df, logger)

    # Drop columnas datetime intermedias: ya están reemplazadas por
    # `dias_desde_*` y no aportan como feature directa al modelo.
    columnas_dt = [c for c in _columnas_maxes() if c in df.columns]
    if columnas_dt:
        df = df.drop(columns=columnas_dt)

    # 4) Guardarraíl: prevalencia del target.
    # DECISIÓN (corrida real sobre dataset full, 2026-05-19): la prevalencia
    # observada es ~57%, no la "minoría" anticipada por el paper. La definición
    # del target coincide exactamente con paper.tex §IV.B (eval >=1 msj, 0
    # opens, 0 clicks), por lo que el desvío es una característica del dataset
    # REES46 para la ventana 2022-10-23 → 2023-04-23, no un bug. Se conserva
    # un guardarraíl en bordes patológicos (≤1% o ≥80%) y se loguea en rango
    # 50-80% como aviso metodológico (clase positiva mayoritaria implica que
    # F1 sobre la clase 1 deja de ser el blanco natural; se reportará también
    # para la clase 0).
    prevalencia = float(df["target_churn_comunicacion"].mean())
    if prevalencia < 0.01 or prevalencia > 0.8:
        logger.error(
            "Prevalencia de churn fuera de rango: %.4f. Revisar definición del target.",
            prevalencia,
        )
        raise RuntimeError(
            f"Prevalencia {prevalencia:.4f} fuera de rango aceptable (1%-80%)."
        )
    if prevalencia > 0.5:
        logger.warning(
            "Prevalencia de churn = %.4f (>50%%). Clase positiva es mayoritaria. "
            "Documentado en PROGRESO.md D7. Pipeline continua.",
            prevalencia,
        )

    # 5) Persistir.
    _persistir(df, logger)

    elapsed = time.time() - t0
    imprimir_checkpoint(
        logger,
        "Fase 2 - Feature engineering",
        {
            "Usuarios evaluables": f"{len(df):,}",
            "Features por usuario": df.shape[1] - 1,
            "Prevalencia target (churn)": f"{prevalencia:.4f}",
            "Parquet generado": PATH_FEATURES,
            "Tiempo total (s)": round(elapsed, 1),
            "Siguiente fase": "Fase 3 - src/train.py",
        },
    )


if __name__ == "__main__":
    ejecutar_fase_2()
