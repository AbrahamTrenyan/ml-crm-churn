"""Pipeline Phase 4: metrics, figures and results report.

Loads the models persisted by Phase 3 (the most recent by timestamp),
applies them to the holdout, computes metrics with 95% bootstrap CIs,
generates PNG figures and writes `outputs/report.md` with native Markdown
tables.

Key technical decisions:

- **Threshold per model**: both the default threshold (0.5) and the
  "optimal F1" one are reported (paper sec. IV.D discusses the confusion
  matrix and recall prioritization; both are reported for transparency).
- **Bootstrap**: `config.bootstrap_iters` resamples with replacement over
  (y_test, y_pred_proba). 95% CI by percentiles (2.5%, 97.5%).
- **Feature importance**: RF and XGB only (paper sec. IV.D). LR is not
  plotted (only its AUC is reported in the comparison table).
- **Tables in `report.md`**: native Markdown, never PNG.
- **Figures**: ROC overlay, PR overlay, per-model confusion matrix,
  feature importance for RF and XGB.

Run standalone:
    python -m src.evaluate
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Headless backend: required in a batch pipeline.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
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
    FIGURES_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    RUN_TIMESTAMP,
    load_config,
    print_checkpoint,
    setup_logger,
)


REPORT_PATH = PROJECT_ROOT / "outputs" / "report.md"

READABLE_MODEL_NAMES = {
    "logreg": "Logistic Regression",
    "rf": "Random Forest",
    "xgb": "XGBoost",
}


# ----------------------------------------------------------------------------
# Model and split loading
# ----------------------------------------------------------------------------


def _latest_file(prefix: str) -> Path:
    """Returns the most recent .pkl with the given prefix in MODELS_DIR."""
    candidates = sorted(MODELS_DIR.glob(f"{prefix}_*.pkl"))
    if not candidates:
        raise FileNotFoundError(
            f"No `{prefix}_*.pkl` files in {MODELS_DIR}. "
            "Run Phase 3 first (python -m src.train)."
        )
    return candidates[-1]


def load_split(logger) -> dict:
    """Loads the most recent train/test split."""
    path = _latest_file("split")
    with open(path, "rb") as f:
        split = pickle.load(f)
    logger.info("Split loaded: %s", path)
    return split


def load_models(logger) -> dict[str, dict]:
    """Loads the 3 models (logreg, rf, xgb), always taking the most recent."""
    out: dict[str, dict] = {}
    for name in ("logreg", "rf", "xgb"):
        path = _latest_file(f"model_{name}")
        with open(path, "rb") as f:
            out[name] = pickle.load(f)
        logger.info("Model %s loaded: %s", name, path)
    return out


# ----------------------------------------------------------------------------
# Metrics and bootstrap
# ----------------------------------------------------------------------------


def _optimal_f1_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Threshold on the PR curve that maximizes F1.

    `precision_recall_curve` returns thresholds of size n-1; F1 may be
    undefined at the extremes: handled with `np.divide(out=zeros)`.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    if len(thresholds) == 0:
        return 0.5
    idx = int(np.argmax(f1[:-1])) if len(f1) > 1 else 0
    return float(thresholds[idx])


def point_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float
) -> dict[str, float]:
    """Scalar metrics for a given threshold + AUC/PR-AUC."""
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
    """95% percentile bootstrap CI for the 5 scalar metrics.

    Resamples (y_true, y_proba) with replacement `n_iters` times and
    recomputes each metric. Returns dict metric -> (low_2.5%, high_97.5%).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    acc: dict[str, list[float]] = {
        k: [] for k in ("auc_roc", "pr_auc", "f1", "precision", "recall")
    }
    for _ in range(n_iters):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_proba[idx]
        # AUC requires both classes present in the resample.
        if len(np.unique(yt)) < 2:
            continue
        yp_bin = (yp >= threshold).astype("int8")
        acc["auc_roc"].append(roc_auc_score(yt, yp))
        acc["pr_auc"].append(average_precision_score(yt, yp))
        acc["f1"].append(f1_score(yt, yp_bin, zero_division=0))
        acc["precision"].append(precision_score(yt, yp_bin, zero_division=0))
        acc["recall"].append(recall_score(yt, yp_bin, zero_division=0))

    return {
        k: (
            float(np.percentile(v, 2.5)) if v else float("nan"),
            float(np.percentile(v, 97.5)) if v else float("nan"),
        )
        for k, v in acc.items()
    }


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------


def roc_overlay_figure(
    results: dict[str, dict], y_true: np.ndarray, logger
) -> Path:
    """Overlaid ROC curves for the 3 models."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_true, r["y_proba"])
        ax.plot(
            fpr,
            tpr,
            label=f"{READABLE_MODEL_NAMES[name]} (AUC={r['default']['auc_roc']:.3f})",
            linewidth=2,
        )
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlabel("False positive rate (FPR)")
    ax.set_ylabel("True positive rate (TPR)")
    ax.set_title("ROC curves comparison")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    path = FIGURES_DIR / f"roc_curves_{RUN_TIMESTAMP}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    logger.info("ROC figure: %s", path)
    return path


def pr_overlay_figure(
    results: dict[str, dict], y_true: np.ndarray, logger
) -> Path:
    """Overlaid Precision-Recall curves."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, r in results.items():
        precision, recall, _ = precision_recall_curve(y_true, r["y_proba"])
        ap = r["default"]["pr_auc"]
        ax.plot(
            recall,
            precision,
            label=f"{READABLE_MODEL_NAMES[name]} (AP={ap:.3f})",
            linewidth=2,
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves comparison")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    path = FIGURES_DIR / f"pr_curves_{RUN_TIMESTAMP}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    logger.info("PR figure: %s", path)
    return path


def confusion_figure(
    name: str, y_true: np.ndarray, y_proba: np.ndarray, threshold: float, logger
) -> Path:
    """Confusion matrix for a model at a given threshold."""
    y_pred = (y_proba >= threshold).astype("int8")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["No churn", "Churn"])
    ax.set_yticks([0, 1], labels=["No churn", "Churn"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(
        f"Confusion matrix - {READABLE_MODEL_NAMES[name]} (thr={threshold:.2f})"
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
    path = FIGURES_DIR / f"confusion_{name}_{RUN_TIMESTAMP}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    logger.info("Confusion figure %s: %s", name, path)
    return path


def feature_importance_figure(
    name: str, artifact: dict, top_k: int, logger
) -> Path | None:
    """Feature importance bar plot for tree-based models."""
    estimator = artifact["estimator"].named_steps["clf"]
    if not hasattr(estimator, "feature_importances_"):
        return None
    imps = pd.Series(
        estimator.feature_importances_,
        index=artifact["feature_names"],
    ).sort_values(ascending=False).head(top_k)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(imps))))
    imps[::-1].plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_k} features - {READABLE_MODEL_NAMES[name]}")
    path = FIGURES_DIR / f"feature_importance_{name}_{RUN_TIMESTAMP}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    logger.info("Feature importance figure %s: %s", name, path)
    return path


# ----------------------------------------------------------------------------
# Markdown report
# ----------------------------------------------------------------------------


def _fmt_metric(value: float, ci: tuple[float, float]) -> str:
    return f"{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def _summary_table_markdown(results: dict[str, dict]) -> str:
    """Comparison table with metrics + 95% CI at the optimal F1 threshold."""
    header = (
        "| Model | AUC-ROC | PR-AUC | F1 | Precision | Recall | Threshold |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for name, r in results.items():
        m = r["optimal_f1"]
        ci = r["ci_optimal_f1"]
        rows.append(
            f"| {READABLE_MODEL_NAMES[name]} "
            f"| {_fmt_metric(m['auc_roc'], ci['auc_roc'])} "
            f"| {_fmt_metric(m['pr_auc'], ci['pr_auc'])} "
            f"| {_fmt_metric(m['f1'], ci['f1'])} "
            f"| {_fmt_metric(m['precision'], ci['precision'])} "
            f"| {_fmt_metric(m['recall'], ci['recall'])} "
            f"| {m['threshold']:.3f} |"
        )
    return header + "\n".join(rows)


def _identify_winner(results: dict[str, dict]) -> str:
    """Model with the best AUC-ROC (the paper prioritizes recall + AUC-ROC)."""
    return max(
        results.items(), key=lambda kv: kv[1]["optimal_f1"]["auc_roc"]
    )[0]


def write_report(
    results: dict[str, dict],
    split: dict,
    figures: dict[str, list[Path]],
    logger,
) -> Path:
    """Writes `outputs/report.md` with methodology, table and embedded figures."""
    config = load_config()
    wd = config["window_dates"]
    winner = _identify_winner(results)

    lines: list[str] = []
    lines.append("# Results report - multichannel CRM communication churn")
    lines.append("")
    lines.append(f"_Generated: {RUN_TIMESTAMP}_")
    lines.append("")

    lines.append("## Methodology (summary for the paper)")
    lines.append("")
    lines.append(
        "- **D1 Time window**: observation = "
        f"{wd['observation_cutoff']} → {wd['evaluation_cutoff']} "
        f"({config['observation_window_months']} months). "
        "Evaluation = "
        f"{wd['evaluation_cutoff']} → {wd['max_sent_at']} "
        f"({config['evaluation_window_months']} months). Counted backwards "
        "from `max(sent_at)`."
    )
    lines.append(
        "- **D2 Purchase attribution**: `is_purchased` is used as it comes "
        "in the dataset (source CRM attribution)."
    )
    lines.append(
        "- **D3 Features**: closed list from paper sec. IV.C. No additional "
        "undocumented features."
    )
    lines.append(
        "- **D4 Environment**: Python 3.11.4, pandas/sklearn/xgboost/imblearn "
        f"stack. Global SEED = {config['seed']}."
    )
    lines.append(
        "- **D5 Dataset**: `messages.csv.gz` (24 months, 721M raw rows), "
        "downloaded from `data.rees46.com`. Processed via gzip streaming + "
        "chunks (never decompressed to disk)."
    )
    lines.append("")

    lines.append("## Split and imbalance")
    lines.append("")
    lines.append(
        f"- Train size: {len(split['y_train']):,} | "
        f"Test size: {len(split['y_test']):,}"
    )
    lines.append(
        f"- Churn prevalence in train: {float(split['y_train'].mean()):.4f}"
    )
    lines.append(
        f"- Churn prevalence in test:  {float(split['y_test'].mean()):.4f}"
    )
    lines.append(
        f"- SMOTE applied: **{split['smote_applied']}** (threshold = "
        f"{config['smote_threshold']})"
    )
    lines.append("")

    lines.append("## Comparative results (optimal F1 threshold)")
    lines.append("")
    lines.append(
        "Metrics with 95% bootstrap confidence intervals "
        f"({config['bootstrap_iters']} iterations)."
    )
    lines.append("")
    lines.append(_summary_table_markdown(results))
    lines.append("")
    lines.append(
        f"**Winning model (best AUC-ROC):** "
        f"{READABLE_MODEL_NAMES[winner]}."
    )
    lines.append("")

    lines.append("## Results at the default threshold (0.5)")
    lines.append("")
    default_header = (
        "| Model | AUC-ROC | PR-AUC | F1 | Precision | Recall |\n"
        "|---|---|---|---|---|---|\n"
    )
    default_rows = []
    for name, r in results.items():
        m = r["default"]
        default_rows.append(
            f"| {READABLE_MODEL_NAMES[name]} "
            f"| {m['auc_roc']:.3f} | {m['pr_auc']:.3f} "
            f"| {m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} |"
        )
    lines.append(default_header + "\n".join(default_rows))
    lines.append("")

    lines.append("## Figures")
    lines.append("")
    for name, paths in figures.items():
        for p in paths:
            rel = p.relative_to(PROJECT_ROOT)
            lines.append(f"![{name}]({rel})")
            lines.append("")

    content = "\n".join(lines) + "\n"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(content, encoding="utf-8")
    logger.info("Report written: %s", REPORT_PATH)
    return REPORT_PATH


# ----------------------------------------------------------------------------
# Phase 4 orchestrator
# ----------------------------------------------------------------------------


def run_phase_4(force: bool = False) -> None:
    """Evaluates the 3 models, generates figures and writes report.md."""
    del force
    logger = setup_logger("phase_4_evaluate")
    t0 = time.time()
    logger.info("=== PHASE 4 START - evaluation ===")

    config = load_config()
    seed = config["seed"]
    n_boot = config["bootstrap_iters"]

    split = load_split(logger)
    models = load_models(logger)
    X_test, y_test = split["X_test"], split["y_test"].to_numpy()

    results: dict[str, dict] = {}
    figures: dict[str, list[Path]] = {}

    for name, artifact in models.items():
        estimator = artifact["estimator"]
        y_proba = estimator.predict_proba(X_test)[:, 1]

        thr_opt = _optimal_f1_threshold(y_test, y_proba)
        m_default = point_metrics(y_test, y_proba, threshold=0.5)
        m_optimal = point_metrics(y_test, y_proba, threshold=thr_opt)
        ci_default = bootstrap_ci(y_test, y_proba, 0.5, n_boot, seed)
        ci_optimal = bootstrap_ci(y_test, y_proba, thr_opt, n_boot, seed)

        results[name] = {
            "y_proba": y_proba,
            "default": m_default,
            "optimal_f1": m_optimal,
            "ci_default": ci_default,
            "ci_optimal_f1": ci_optimal,
        }
        logger.info(
            "%s: AUC=%.4f, PR-AUC=%.4f, F1@thr*=%.4f (thr*=%.3f)",
            name,
            m_optimal["auc_roc"],
            m_optimal["pr_auc"],
            m_optimal["f1"],
            thr_opt,
        )

        # Per-model confusion figure (at the optimal threshold).
        fp_cm = confusion_figure(name, y_test, y_proba, thr_opt, logger)
        figures.setdefault(name, []).append(fp_cm)

        # Feature importance only for tree-based models.
        fp_fi = feature_importance_figure(name, artifact, top_k=20, logger=logger)
        if fp_fi is not None:
            figures.setdefault(name, []).append(fp_fi)

    # Global comparison figures.
    fp_roc = roc_overlay_figure(results, y_test, logger)
    fp_pr = pr_overlay_figure(results, y_test, logger)
    figures["comparison"] = [fp_roc, fp_pr]

    # Final report.
    report_path = write_report(results, split, figures, logger)

    elapsed = time.time() - t0
    print_checkpoint(
        logger,
        "Phase 4 - Evaluation",
        {
            "Models evaluated": list(results.keys()),
            "Winner (AUC-ROC)": _identify_winner(results),
            "Bootstrap iters": n_boot,
            "Figures generated": sum(len(v) for v in figures.values()),
            "Report": report_path,
            "Total time (s)": round(elapsed, 1),
            "Next phase": "(end of pipeline)",
        },
    )


if __name__ == "__main__":
    run_phase_4()
