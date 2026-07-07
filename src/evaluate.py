"""Fase 4 del pipeline: métricas, figuras y reporte académico.

Carga los modelos persistidos por Fase 3 (los más recientes por timestamp),
los aplica al holdout, calcula métricas con CI 95% por bootstrap, genera
figuras PNG y emite `outputs/report.md` con tablas Markdown nativas.

Decisiones técnicas clave:

- **Threshold por modelo**: se reporta tanto el threshold por defecto (0.5)
  como el "óptimo F1" (paper sec. IV.D habla de matriz de confusión y
  priorización de recall; reportamos ambos para transparencia).
- **Bootstrap**: `config.bootstrap_iters` (1000 por defecto) resampleos
  con reemplazo sobre (y_test, y_pred_proba). CI 95% por percentiles
  (2.5%, 97.5%).
- **Feature importance**: solo RF y XGB (paper sec. IV.D línea 304). LR
  no se grafica (solo se reporta su AUC en la tabla comparativa).
- **Tablas en `report.md`**: Markdown nativo, nunca PNG.
- **Figuras**: ROC overlay, PR overlay, confusión por modelo,
  feature importance para RF y XGB.

Ejecutar standalone:
    python -m src.evaluate
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Backend sin display: necesario en pipeline batch.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.logging_utils import (
    DIR_FIGURES,
    DIR_MODELS,
    RAIZ_PROYECTO,
    TIMESTAMP_CORRIDA,
    cargar_config,
    configurar_logger,
    imprimir_checkpoint,
)


PATH_REPORT = RAIZ_PROYECTO / "outputs" / "report.md"

NOMBRES_MODELOS_LEGIBLES = {
    "logreg": "Logistic Regression",
    "rf": "Random Forest",
    "xgb": "XGBoost",
}


# ----------------------------------------------------------------------------
# Carga de modelos y split
# ----------------------------------------------------------------------------


def _ultimo_archivo(prefijo: str) -> Path:
    """Devuelve el .pkl más reciente con el prefijo dado en DIR_MODELS."""
    candidatos = sorted(DIR_MODELS.glob(f"{prefijo}_*.pkl"))
    if not candidatos:
        raise FileNotFoundError(
            f"No hay archivos `{prefijo}_*.pkl` en {DIR_MODELS}. "
            "Correr antes Fase 3 (python -m src.train)."
        )
    return candidatos[-1]


def cargar_split(logger) -> dict:
    """Carga el split de train/test más reciente."""
    path = _ultimo_archivo("split")
    with open(path, "rb") as f:
        split = pickle.load(f)
    logger.info("Split cargado: %s", path)
    return split


def cargar_modelos(logger) -> dict[str, dict]:
    """Carga los 3 modelos (logreg, rf, xgb), tomando siempre el más reciente."""
    out: dict[str, dict] = {}
    for nombre in ("logreg", "rf", "xgb"):
        path = _ultimo_archivo(f"modelo_{nombre}")
        with open(path, "rb") as f:
            out[nombre] = pickle.load(f)
        logger.info("Modelo %s cargado: %s", nombre, path)
    return out


# ----------------------------------------------------------------------------
# Métricas y bootstrap
# ----------------------------------------------------------------------------


def _threshold_optimo_f1(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Threshold sobre la curva PR que maximiza F1.

    `precision_recall_curve` devuelve thresholds de tamaño n-1; F1 puede
    indefinirse en extremos: lo manejamos con `np.divide(out=zeros)`.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    if len(thresholds) == 0:
        return 0.5
    idx = int(np.argmax(f1[:-1])) if len(f1) > 1 else 0
    return float(thresholds[idx])


def metricas_punto(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float
) -> dict[str, float]:
    """Métricas escalares para un threshold dado + AUC/PR-AUC."""
    y_pred = (y_proba >= threshold).astype("int8")
    return {
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    n_iters: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Bootstrap percentil 95% CI para las 5 métricas escalares.

    Resamplea (y_true, y_proba) con reemplazo `n_iters` veces y recalcula
    cada métrica. Devuelve dict metrica -> (low_2.5%, high_97.5%).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    acum: dict[str, list[float]] = {
        k: [] for k in ("auc_roc", "pr_auc", "f1", "precision", "recall")
    }
    for _ in range(n_iters):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_proba[idx]
        # AUC requiere ambas clases presentes en el resample.
        if len(np.unique(yt)) < 2:
            continue
        yp_bin = (yp >= threshold).astype("int8")
        acum["auc_roc"].append(roc_auc_score(yt, yp))
        acum["pr_auc"].append(average_precision_score(yt, yp))
        acum["f1"].append(f1_score(yt, yp_bin, zero_division=0))
        acum["precision"].append(precision_score(yt, yp_bin, zero_division=0))
        acum["recall"].append(recall_score(yt, yp_bin, zero_division=0))

    return {
        k: (
            float(np.percentile(v, 2.5)) if v else float("nan"),
            float(np.percentile(v, 97.5)) if v else float("nan"),
        )
        for k, v in acum.items()
    }


# ----------------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------------


def figura_roc_overlay(
    resultados: dict[str, dict], y_true: np.ndarray, logger
) -> Path:
    """Plot de curvas ROC superpuestas para los 3 modelos."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for nombre, r in resultados.items():
        fpr, tpr, _ = roc_curve(y_true, r["y_proba"])
        ax.plot(
            fpr,
            tpr,
            label=f"{NOMBRES_MODELOS_LEGIBLES[nombre]} (AUC={r['default']['auc_roc']:.3f})",
            linewidth=2,
        )
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlabel("Tasa de falsos positivos (FPR)")
    ax.set_ylabel("Tasa de verdaderos positivos (TPR)")
    ax.set_title("Curvas ROC comparativas")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    path = DIR_FIGURES / f"roc_curves_{TIMESTAMP_CORRIDA}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    logger.info("Figura ROC: %s", path)
    return path


def figura_pr_overlay(
    resultados: dict[str, dict], y_true: np.ndarray, logger
) -> Path:
    """Plot de curvas Precision-Recall superpuestas."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for nombre, r in resultados.items():
        precision, recall, _ = precision_recall_curve(y_true, r["y_proba"])
        ap = r["default"]["pr_auc"]
        ax.plot(
            recall,
            precision,
            label=f"{NOMBRES_MODELOS_LEGIBLES[nombre]} (AP={ap:.3f})",
            linewidth=2,
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curvas Precision-Recall comparativas")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    path = DIR_FIGURES / f"pr_curves_{TIMESTAMP_CORRIDA}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    logger.info("Figura PR: %s", path)
    return path


def figura_confusion(
    nombre: str, y_true: np.ndarray, y_proba: np.ndarray, threshold: float, logger
) -> Path:
    """Matriz de confusión para un modelo y threshold dado."""
    y_pred = (y_proba >= threshold).astype("int8")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["No churn", "Churn"])
    ax.set_yticks([0, 1], labels=["No churn", "Churn"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(
        f"Matriz de confusión - {NOMBRES_MODELOS_LEGIBLES[nombre]} (thr={threshold:.2f})"
    )
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=11,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path = DIR_FIGURES / f"confusion_{nombre}_{TIMESTAMP_CORRIDA}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    logger.info("Figura confusión %s: %s", nombre, path)
    return path


def figura_feature_importance(
    nombre: str, artefacto: dict, top_k: int, logger
) -> Path | None:
    """Bar plot de feature importance para modelos basados en árboles."""
    estimador = artefacto["estimador"].named_steps["clf"]
    if not hasattr(estimador, "feature_importances_"):
        return None
    imps = pd.Series(
        estimador.feature_importances_,
        index=artefacto["feature_names"],
    ).sort_values(ascending=False).head(top_k)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(imps))))
    imps[::-1].plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Importancia")
    ax.set_title(f"Top {top_k} features - {NOMBRES_MODELOS_LEGIBLES[nombre]}")
    path = DIR_FIGURES / f"feature_importance_{nombre}_{TIMESTAMP_CORRIDA}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    logger.info("Figura feature_importance %s: %s", nombre, path)
    return path


# ----------------------------------------------------------------------------
# Reporte Markdown
# ----------------------------------------------------------------------------


def _fmt_metric(valor: float, ci: tuple[float, float]) -> str:
    return f"{valor:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def _tabla_resumen_markdown(resultados: dict[str, dict]) -> str:
    """Tabla comparativa con métricas + CI 95% sobre el threshold óptimo F1."""
    cabecera = (
        "| Modelo | AUC-ROC | PR-AUC | F1 | Precision | Recall | Threshold |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    filas = []
    for nombre, r in resultados.items():
        m = r["optimo_f1"]
        ci = r["ci_optimo_f1"]
        filas.append(
            f"| {NOMBRES_MODELOS_LEGIBLES[nombre]} "
            f"| {_fmt_metric(m['auc_roc'], ci['auc_roc'])} "
            f"| {_fmt_metric(m['pr_auc'], ci['pr_auc'])} "
            f"| {_fmt_metric(m['f1'], ci['f1'])} "
            f"| {_fmt_metric(m['precision'], ci['precision'])} "
            f"| {_fmt_metric(m['recall'], ci['recall'])} "
            f"| {m['threshold']:.3f} |"
        )
    return cabecera + "\n".join(filas)


def _identificar_ganador(resultados: dict[str, dict]) -> str:
    """Modelo con mejor AUC-ROC (paper prioriza recall + AUC-ROC)."""
    return max(
        resultados.items(), key=lambda kv: kv[1]["optimo_f1"]["auc_roc"]
    )[0]


def escribir_reporte(
    resultados: dict[str, dict],
    split: dict,
    figuras: dict[str, list[Path]],
    logger,
) -> Path:
    """Escribe `outputs/report.md` con metodología, tabla y figuras embebidas."""
    config = cargar_config()
    vf = config["ventana_fechas"]
    ganador = _identificar_ganador(resultados)

    lineas: list[str] = []
    lineas.append("# Reporte de resultados - churn de comunicación CRM multicanal")
    lineas.append("")
    lineas.append(f"_Generado: {TIMESTAMP_CORRIDA}_")
    lineas.append("")

    lineas.append("## Metodología (síntesis para el paper)")
    lineas.append("")
    lineas.append(
        "- **D1 Ventana temporal**: observación = "
        f"{vf['corte_observacion']} → {vf['corte_evaluacion']} "
        f"({config['ventana_observacion_meses']} meses). "
        "Evaluación = "
        f"{vf['corte_evaluacion']} → {vf['max_sent_at']} "
        f"({config['ventana_evaluacion_meses']} meses). Conteo hacia atrás "
        "desde `max(sent_at)`."
    )
    lineas.append(
        "- **D2 Atribución de compra**: se usa `is_purchased` del dataset "
        "tal como viene (atribución CRM origen)."
    )
    lineas.append(
        "- **D3 Features**: lista cerrada del paper sec. IV.C. Sin features "
        "adicionales no documentadas."
    )
    lineas.append(
        "- **D4 Entorno**: Python 3.11.4, stack pandas/sklearn/xgboost/imblearn. "
        f"SEED = {config['seed']} global."
    )
    lineas.append(
        "- **D5 Dataset**: `messages.csv.gz` (24 meses, 721M filas en bruto), "
        "descargado de `data.rees46.com`. Se procesa por streaming gzip + "
        "chunks (sin descomprimir a disco)."
    )
    lineas.append("")

    lineas.append("## Split y desbalance")
    lineas.append("")
    lineas.append(
        f"- Tamaño train: {len(split['y_train']):,} | "
        f"Tamaño test: {len(split['y_test']):,}"
    )
    lineas.append(
        f"- Prevalencia churn en train: {float(split['y_train'].mean()):.4f}"
    )
    lineas.append(
        f"- Prevalencia churn en test:  {float(split['y_test'].mean()):.4f}"
    )
    lineas.append(
        f"- SMOTE aplicado: **{split['aplicar_smote']}** (threshold = "
        f"{config['smote_threshold']})"
    )
    lineas.append("")

    lineas.append("## Resultados comparativos (threshold óptimo F1)")
    lineas.append("")
    lineas.append(
        "Métricas con intervalo de confianza 95% por bootstrap "
        f"({config['bootstrap_iters']} iteraciones)."
    )
    lineas.append("")
    lineas.append(_tabla_resumen_markdown(resultados))
    lineas.append("")
    lineas.append(
        f"**Modelo ganador (mejor AUC-ROC):** "
        f"{NOMBRES_MODELOS_LEGIBLES[ganador]}."
    )
    lineas.append("")

    lineas.append("## Resultados con threshold por defecto (0.5)")
    lineas.append("")
    cabecera_def = (
        "| Modelo | AUC-ROC | PR-AUC | F1 | Precision | Recall |\n"
        "|---|---|---|---|---|---|\n"
    )
    filas_def = []
    for nombre, r in resultados.items():
        m = r["default"]
        filas_def.append(
            f"| {NOMBRES_MODELOS_LEGIBLES[nombre]} "
            f"| {m['auc_roc']:.3f} | {m['pr_auc']:.3f} "
            f"| {m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} |"
        )
    lineas.append(cabecera_def + "\n".join(filas_def))
    lineas.append("")

    lineas.append("## Figuras")
    lineas.append("")
    for nombre, paths in figuras.items():
        for p in paths:
            rel = p.relative_to(RAIZ_PROYECTO)
            lineas.append(f"![{nombre}]({rel})")
            lineas.append("")

    contenido = "\n".join(lineas) + "\n"
    PATH_REPORT.parent.mkdir(parents=True, exist_ok=True)
    PATH_REPORT.write_text(contenido, encoding="utf-8")
    logger.info("Reporte escrito: %s", PATH_REPORT)
    return PATH_REPORT


# ----------------------------------------------------------------------------
# Orquestador de Fase 4
# ----------------------------------------------------------------------------


def ejecutar_fase_4(force: bool = False) -> None:
    """Evalúa los 3 modelos, genera figuras y escribe report.md."""
    del force
    logger = configurar_logger("fase_4_evaluate")
    t0 = time.time()
    logger.info("=== INICIO FASE 4 - evaluacion ===")

    config = cargar_config()
    seed = config["seed"]
    n_boot = config["bootstrap_iters"]

    split = cargar_split(logger)
    modelos = cargar_modelos(logger)
    X_test, y_test = split["X_test"], split["y_test"].to_numpy()

    resultados: dict[str, dict] = {}
    figuras: dict[str, list[Path]] = {}

    for nombre, artefacto in modelos.items():
        estimador = artefacto["estimador"]
        y_proba = estimador.predict_proba(X_test)[:, 1]

        thr_opt = _threshold_optimo_f1(y_test, y_proba)
        m_default = metricas_punto(y_test, y_proba, threshold=0.5)
        m_optimo = metricas_punto(y_test, y_proba, threshold=thr_opt)
        ci_default = bootstrap_ci(y_test, y_proba, 0.5, n_boot, seed)
        ci_optimo = bootstrap_ci(y_test, y_proba, thr_opt, n_boot, seed)

        resultados[nombre] = {
            "y_proba": y_proba,
            "default": m_default,
            "optimo_f1": m_optimo,
            "ci_default": ci_default,
            "ci_optimo_f1": ci_optimo,
        }
        logger.info(
            "%s: AUC=%.4f, PR-AUC=%.4f, F1@thr*=%.4f (thr*=%.3f)",
            nombre,
            m_optimo["auc_roc"],
            m_optimo["pr_auc"],
            m_optimo["f1"],
            thr_opt,
        )

        # Figura de confusión por modelo (al threshold óptimo).
        fp_cm = figura_confusion(nombre, y_test, y_proba, thr_opt, logger)
        figuras.setdefault(nombre, []).append(fp_cm)

        # Feature importance solo para modelos basados en árboles.
        fp_fi = figura_feature_importance(nombre, artefacto, top_k=20, logger=logger)
        if fp_fi is not None:
            figuras.setdefault(nombre, []).append(fp_fi)

    # Figuras comparativas globales.
    fp_roc = figura_roc_overlay(resultados, y_test, logger)
    fp_pr = figura_pr_overlay(resultados, y_test, logger)
    figuras["comparativas"] = [fp_roc, fp_pr]

    # Reporte final.
    path_report = escribir_reporte(resultados, split, figuras, logger)

    elapsed = time.time() - t0
    imprimir_checkpoint(
        logger,
        "Fase 4 - Evaluación",
        {
            "Modelos evaluados": list(resultados.keys()),
            "Ganador (AUC-ROC)": _identificar_ganador(resultados),
            "Bootstrap iters": n_boot,
            "Figuras generadas": sum(len(v) for v in figuras.values()),
            "Reporte": path_report,
            "Tiempo total (s)": round(elapsed, 1),
            "Siguiente fase": "(fin del pipeline)",
        },
    )


if __name__ == "__main__":
    ejecutar_fase_4()
