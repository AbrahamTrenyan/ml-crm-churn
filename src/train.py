"""Fase 3 del pipeline: entrenamiento de los 3 modelos.

Entrena Logistic Regression (baseline), Random Forest y XGBoost sobre
`outputs/features/features_por_usuario.parquet` y persiste cada modelo
ajustado (junto con metadatos del split y la búsqueda) en
`outputs/models/modelo_<nombre>_<timestamp>.pkl`.

Decisiones técnicas clave (todas en línea con CLAUDE.md y PROGRESO.md):

- **Split principal**: train/test estratificado 80/20 sobre el universo de
  usuarios evaluables. El split temporal "fuerte" del pipeline ya está
  garantizado en Fase 2 (features sobre obs, target sobre eval).
- **Cross-validation**: `StratifiedKFold(n_splits=cv_folds)` (paper sec. IV.D
  línea 302 dice "validación cruzada estratificada de cinco pliegues").
- **SMOTE condicional**: si la prevalencia positiva en TRAIN es < 0.2 (config:
  `smote_threshold`), se inserta SMOTE en el pipeline. Estrictamente vía
  `imblearn.pipeline.Pipeline` para que SMOTE actúe solo en el split de
  entrenamiento de cada fold (no en validación).
- **XGB + SMOTE**: si SMOTE se activa, `scale_pos_weight=1` (no se acumulan
  dos correcciones de desbalance).
- **Random search**: 20 iters (config: `random_search_n_iter`), optimizando
  AUC-ROC. LR no se busca: regularización por defecto.
- **Subsample para búsqueda (D8)**: con el dataset full (9.9M usuarios), correr
  100 fits × 6.3M samples no es viable (24-44 h estimadas). Se toma un
  subsample estratificado de `subsample_busqueda` (default 500.000) sobre
  X_train/y_train para alimentar `RandomizedSearchCV`. Una vez identificados
  los mejores hiperparámetros, se hace `.fit` final del pipeline ganador
  sobre **TODO** `X_train` (~7.9M) → el modelo persistido aprovecha la
  totalidad del dataset. LR sí se entrena directamente sobre el train completo
  (no requiere búsqueda).
- **Standardización**: solo para LR. RF/XGB se entrenan sobre las features
  originales (paper sec. IV.D línea 300).
- **Imputación**: las features de tasas/recencias pueden ser NaN
  (usuarios sin opens, sin canal, etc.). Se imputan con la mediana del train.
- **Seed global**: SEED=42 en todo splitter/estimador (config).

Ejecutar standalone:
    python -m src.train
"""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features import PATH_FEATURES
from src.logging_utils import (
    DIR_MODELS,
    TIMESTAMP_CORRIDA,
    cargar_config,
    configurar_logger,
    imprimir_checkpoint,
)


# ----------------------------------------------------------------------------
# Tipos auxiliares
# ----------------------------------------------------------------------------


@dataclass
class DatosSplit:
    """Contenedor del split train/test ya separado por features/target."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    @property
    def prevalencia_train(self) -> float:
        return float(self.y_train.mean())


# ----------------------------------------------------------------------------
# Carga y split
# ----------------------------------------------------------------------------


COLUMNAS_NO_FEATURE = {
    "client_id",
    "target_churn_comunicacion",
    "n_msgs_eval",
    "n_opens_eval",
    "n_clicks_eval",
}


def _features_y_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa el DataFrame en X (features numéricas) y target."""
    y = df["target_churn_comunicacion"].astype("int8")
    columnas_feat = [c for c in df.columns if c not in COLUMNAS_NO_FEATURE]
    X = df[columnas_feat].copy()
    # Datetime columns (ultimo_open/click_*) NO van como features directas:
    # ya se transformaron en `dias_desde_ultimo_*` en Fase 2. Las eliminamos
    # explícitamente por si quedaron en el parquet.
    X = X.select_dtypes(exclude=["datetime64[ns]", "datetime64"])
    return X, y


def preparar_split(df: pd.DataFrame, seed: int, logger) -> DatosSplit:
    """Train/test estratificado 80/20."""
    X, y = _features_y_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    logger.info(
        "Split listo: train=%d (prev=%.4f), test=%d (prev=%.4f), features=%d",
        len(X_train),
        float(y_train.mean()),
        len(X_test),
        float(y_test.mean()),
        X_train.shape[1],
    )
    return DatosSplit(X_train, X_test, y_train, y_test)


# ----------------------------------------------------------------------------
# Pipelines
# ----------------------------------------------------------------------------


def _pasos_preproc(estandarizar: bool) -> list[tuple[str, Any]]:
    """Pasos comunes de preprocesamiento: imputación + (opcional) escalado."""
    pasos: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if estandarizar:
        pasos.append(("scaler", StandardScaler()))
    return pasos


def _pipeline_con_smote(
    pasos_preproc: list[tuple[str, Any]],
    estimador: Any,
    aplicar_smote: bool,
    seed: int,
) -> ImbPipeline:
    """Arma un imblearn Pipeline con SMOTE opcional antes del estimador.

    SMOTE va DESPUÉS del preproc para no introducir NaNs por el escalado,
    y ANTES del estimador. imblearn aplica SMOTE solo durante .fit (no
    en .transform/.predict ni durante validación cruzada en el fold de val).
    """
    pasos = list(pasos_preproc)
    if aplicar_smote:
        pasos.append(("smote", SMOTE(random_state=seed)))
    pasos.append(("clf", estimador))
    return ImbPipeline(pasos)


# ----------------------------------------------------------------------------
# Definiciones de modelos + búsquedas
# ----------------------------------------------------------------------------


def construir_pipeline_logreg(aplicar_smote: bool, seed: int) -> ImbPipeline:
    """LR con regularización por defecto (paper: baseline sin búsqueda)."""
    pasos = _pasos_preproc(estandarizar=True)
    return _pipeline_con_smote(
        pasos,
        LogisticRegression(max_iter=1000, random_state=seed),
        aplicar_smote=aplicar_smote,
        seed=seed,
    )


def construir_pipeline_rf(aplicar_smote: bool, seed: int) -> ImbPipeline:
    pasos = _pasos_preproc(estandarizar=False)
    return _pipeline_con_smote(
        pasos,
        RandomForestClassifier(random_state=seed, n_jobs=-1),
        aplicar_smote=aplicar_smote,
        seed=seed,
    )


def construir_pipeline_xgb(aplicar_smote: bool, seed: int) -> ImbPipeline:
    pasos = _pasos_preproc(estandarizar=False)
    # DECISIÓN: si SMOTE está activo, scale_pos_weight=1 (CLAUDE.md). Si no,
    # dejamos scale_pos_weight como hiperparámetro de la búsqueda.
    xgb_kwargs = dict(
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="auc",
    )
    if aplicar_smote:
        xgb_kwargs["scale_pos_weight"] = 1
    return _pipeline_con_smote(
        pasos,
        XGBClassifier(**xgb_kwargs),
        aplicar_smote=aplicar_smote,
        seed=seed,
    )


def _grid_rf() -> dict:
    """Espacio acotado de hiperparámetros para RF (random search 20 iters)."""
    return {
        "clf__n_estimators": [200, 400, 600, 800],
        "clf__max_depth": [None, 8, 16, 24],
        "clf__min_samples_split": [2, 5, 10, 20],
        "clf__min_samples_leaf": [1, 2, 4, 8],
        "clf__max_features": ["sqrt", "log2"],
    }


def _grid_xgb(aplicar_smote: bool) -> dict:
    """Espacio acotado de hiperparámetros para XGBoost."""
    grid = {
        "clf__n_estimators": [200, 400, 600, 800],
        "clf__max_depth": [4, 6, 8, 10],
        "clf__learning_rate": [0.03, 0.05, 0.1, 0.2],
        "clf__subsample": [0.7, 0.8, 1.0],
        "clf__colsample_bytree": [0.7, 0.8, 1.0],
        "clf__reg_lambda": [0.5, 1.0, 2.0],
    }
    if not aplicar_smote:
        # Sin SMOTE, sí tunear scale_pos_weight para abordar desbalance.
        grid["clf__scale_pos_weight"] = [1.0, 3.0, 5.0, 10.0]
    return grid


def _subsample_estratificado(
    X: pd.DataFrame,
    y: pd.Series,
    n_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Subsample estratificado por target. Si `n_samples >= len(X)`, devuelve todo."""
    if n_samples >= len(X):
        return X, y
    # Usamos train_test_split con stratify=y para obtener una particion
    # estratificada del tamaño pedido (la "train" del split que tiramos).
    X_sub, _, y_sub, _ = train_test_split(
        X, y,
        train_size=n_samples,
        random_state=seed,
        stratify=y,
    )
    return X_sub, y_sub


def correr_random_search(
    pipeline: ImbPipeline,
    grid: dict,
    datos: DatosSplit,
    n_iter: int,
    cv_folds: int,
    seed: int,
    logger,
    subsample_size: int | None = None,
) -> RandomizedSearchCV:
    """Random search con CV estratificada, optimizando AUC-ROC.

    Si `subsample_size` se proporciona y es menor que el tamaño de train,
    la búsqueda se hace sobre un subsample estratificado. Esto NO afecta la
    semántica de los hiperparámetros encontrados: solo acelera la búsqueda.
    El refit sobre el dataset completo se hace fuera de esta función con
    `_refit_sobre_full_train`.
    """
    if subsample_size is not None and subsample_size < len(datos.X_train):
        X_busqueda, y_busqueda = _subsample_estratificado(
            datos.X_train, datos.y_train, subsample_size, seed
        )
        logger.info(
            "Subsample para busqueda: %d (de %d train), prevalencia=%.4f",
            len(X_busqueda),
            len(datos.X_train),
            float(y_busqueda.mean()),
        )
    else:
        X_busqueda, y_busqueda = datos.X_train, datos.y_train

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=grid,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,  # estimadores ya usan n_jobs=-1 internamente
        random_state=seed,
        # No refiteamos acá sobre X_busqueda: vamos a refittear afuera
        # sobre el train completo con los best_params encontrados.
        refit=False,
        return_train_score=False,
        verbose=1,
    )
    t0 = time.time()
    search.fit(X_busqueda, y_busqueda)
    elapsed = time.time() - t0
    logger.info(
        "Random search OK: best_auc=%.4f, params=%s, %.1fs",
        search.best_score_,
        search.best_params_,
        elapsed,
    )
    return search


def _refit_sobre_full_train(
    pipeline: ImbPipeline,
    best_params: dict,
    datos: DatosSplit,
    logger,
) -> ImbPipeline:
    """Aplica `best_params` al pipeline y lo fittea sobre TODO X_train.

    Patron D8: la búsqueda corre sobre un subsample (rápido), el modelo
    final aprovecha el train completo (~7.9M) para maximizar señal.
    """
    pipeline = pipeline.set_params(**best_params)
    t0 = time.time()
    pipeline.fit(datos.X_train, datos.y_train)
    logger.info(
        "Refit sobre full train (%d samples) OK en %.1fs",
        len(datos.X_train),
        time.time() - t0,
    )
    return pipeline


# ----------------------------------------------------------------------------
# Persistencia de modelos
# ----------------------------------------------------------------------------


def _path_modelo(nombre: str) -> str:
    DIR_MODELS.mkdir(parents=True, exist_ok=True)
    return str(DIR_MODELS / f"modelo_{nombre}_{TIMESTAMP_CORRIDA}.pkl")


def _persistir_modelo(
    nombre: str,
    estimador: Any,
    datos: DatosSplit,
    aplicar_smote: bool,
    best_params: dict | None,
    logger,
) -> str:
    """Serializa el modelo entrenado + metadatos del split en un solo .pkl."""
    artefacto = {
        "nombre": nombre,
        "estimador": estimador,
        "feature_names": list(datos.X_train.columns),
        "aplicar_smote": aplicar_smote,
        "best_params": best_params,
        "timestamp": TIMESTAMP_CORRIDA,
        "n_train": len(datos.X_train),
        "n_test": len(datos.X_test),
        "prevalencia_train": datos.prevalencia_train,
    }
    path = _path_modelo(nombre)
    with open(path, "wb") as f:
        pickle.dump(artefacto, f)
    logger.info("Modelo persistido: %s", path)
    return path


# ----------------------------------------------------------------------------
# Orquestador de Fase 3
# ----------------------------------------------------------------------------


def ejecutar_fase_3(force: bool = False) -> dict:
    """Entrena los 3 modelos y devuelve un dict con paths de artefactos.

    El parámetro `force` no cambia el cómputo (no hay caching: cada run produce
    nuevos pkl con timestamp). Se acepta por consistencia con la CLI de run.py.
    """
    del force  # actualmente sin efecto; ver docstring
    logger = configurar_logger("fase_3_train")
    t0 = time.time()
    logger.info("=== INICIO FASE 3 - entrenamiento ===")

    config = cargar_config()
    seed = config["seed"]
    smote_threshold = config["smote_threshold"]
    n_iter = config["random_search_n_iter"]
    cv_folds = config["cv_folds"]
    subsample_busqueda = config.get("subsample_busqueda")

    # 1) Cargar features.
    if not PATH_FEATURES.exists():
        raise RuntimeError(
            f"No existe {PATH_FEATURES}. Correr antes Fase 2 (python -m src.features)."
        )
    df = pd.read_parquet(PATH_FEATURES)
    logger.info("Features cargadas: %d usuarios, %d columnas", *df.shape)

    # 2) Split estratificado y decisión SMOTE.
    datos = preparar_split(df, seed=seed, logger=logger)
    aplicar_smote = datos.prevalencia_train < smote_threshold
    logger.info(
        "Prevalencia train=%.4f, threshold=%.2f -> SMOTE=%s",
        datos.prevalencia_train,
        smote_threshold,
        aplicar_smote,
    )

    artefactos: dict[str, str] = {}

    # 3) Logistic Regression (sin búsqueda, fit directo).
    logger.info("--- Entrenando Logistic Regression (baseline) ---")
    pipe_lr = construir_pipeline_logreg(aplicar_smote=aplicar_smote, seed=seed)
    t_lr = time.time()
    pipe_lr.fit(datos.X_train, datos.y_train)
    logger.info("LR entrenado en %.1fs", time.time() - t_lr)
    artefactos["logreg"] = _persistir_modelo(
        "logreg", pipe_lr, datos, aplicar_smote, best_params=None, logger=logger
    )

    # 4) Random Forest con random search (sobre subsample) + refit sobre full.
    logger.info(
        "--- Entrenando Random Forest (random search %d iters, subsample=%s) ---",
        n_iter, subsample_busqueda,
    )
    pipe_rf = construir_pipeline_rf(aplicar_smote=aplicar_smote, seed=seed)
    search_rf = correr_random_search(
        pipe_rf, _grid_rf(), datos, n_iter=n_iter, cv_folds=cv_folds,
        seed=seed, logger=logger, subsample_size=subsample_busqueda,
    )
    pipe_rf_final = _refit_sobre_full_train(pipe_rf, search_rf.best_params_, datos, logger)
    artefactos["rf"] = _persistir_modelo(
        "rf", pipe_rf_final, datos, aplicar_smote,
        best_params=search_rf.best_params_, logger=logger,
    )

    # 5) XGBoost con random search (sobre subsample) + refit sobre full.
    logger.info(
        "--- Entrenando XGBoost (random search %d iters, subsample=%s) ---",
        n_iter, subsample_busqueda,
    )
    pipe_xgb = construir_pipeline_xgb(aplicar_smote=aplicar_smote, seed=seed)
    search_xgb = correr_random_search(
        pipe_xgb, _grid_xgb(aplicar_smote), datos,
        n_iter=n_iter, cv_folds=cv_folds, seed=seed, logger=logger,
        subsample_size=subsample_busqueda,
    )
    pipe_xgb_final = _refit_sobre_full_train(pipe_xgb, search_xgb.best_params_, datos, logger)
    artefactos["xgb"] = _persistir_modelo(
        "xgb", pipe_xgb_final, datos, aplicar_smote,
        best_params=search_xgb.best_params_, logger=logger,
    )

    # 6) Persistir el split (para que evaluate.py use exactamente el mismo).
    split_path = DIR_MODELS / f"split_{TIMESTAMP_CORRIDA}.pkl"
    with open(split_path, "wb") as f:
        pickle.dump(
            {
                "X_train": datos.X_train,
                "X_test": datos.X_test,
                "y_train": datos.y_train,
                "y_test": datos.y_test,
                "seed": seed,
                "aplicar_smote": aplicar_smote,
                "timestamp": TIMESTAMP_CORRIDA,
            },
            f,
        )
    artefactos["split"] = str(split_path)
    logger.info("Split persistido en %s", split_path)

    elapsed = time.time() - t0
    imprimir_checkpoint(
        logger,
        "Fase 3 - Entrenamiento",
        {
            "Modelos entrenados": "logreg, rf, xgb",
            "SMOTE aplicado": aplicar_smote,
            "Tamaño train": f"{len(datos.X_train):,}",
            "Tamaño test": f"{len(datos.X_test):,}",
            "Features": datos.X_train.shape[1],
            "Best AUC (CV) RF": f"{search_rf.best_score_:.4f}",
            "Best AUC (CV) XGB": f"{search_xgb.best_score_:.4f}",
            "Artefactos": artefactos,
            "Tiempo total (s)": round(elapsed, 1),
            "Siguiente fase": "Fase 4 - src/evaluate.py",
        },
    )
    return artefactos


if __name__ == "__main__":
    ejecutar_fase_3()
