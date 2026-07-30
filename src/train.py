"""Pipeline Phase 3: training of the 3 models.

Trains Logistic Regression (baseline), Random Forest and XGBoost over
`outputs/features/features_per_user.parquet` and persists each fitted model
(along with split and search metadata) to
`outputs/models/model_<name>_<timestamp>.pkl`.

Key technical decisions:

- **Main split**: stratified 80/20 train/test over the universe of
  evaluable users. The pipeline's "strong" temporal split is already
  guaranteed in Phase 2 (features over obs, target over eval).
- **Cross-validation**: `StratifiedKFold(n_splits=cv_folds)` (paper sec.
  IV.D: "stratified five-fold cross-validation").
- **Conditional SMOTE**: if the positive prevalence in TRAIN is < 0.2
  (config: `smote_threshold`), SMOTE is inserted into the pipeline.
  Strictly via `imblearn.pipeline.Pipeline` so SMOTE acts only on the
  training split of each fold (never on validation).
- **XGB + SMOTE**: if SMOTE is active, `scale_pos_weight=1` (two imbalance
  corrections are never stacked).
- **Random search**: 20 iters (config: `random_search_n_iter`), optimizing
  AUC-ROC. LR is not searched: default regularization.
- **Search subsample (D8)**: with the full dataset (9.9M users), running
  100 fits x 6.3M samples is not viable (24-44 h estimated). A stratified
  subsample of `search_subsample` (default 500,000) is drawn from
  X_train/y_train to feed `RandomizedSearchCV`. Once the best
  hyperparameters are identified, the winning pipeline gets a final `.fit`
  over the WHOLE `X_train` (~7.9M) -> the persisted model leverages the
  entire dataset. LR is trained directly on the full train (no search
  needed).
- **Standardization**: only for LR. RF/XGB are trained on the original
  features (paper sec. IV.D).
- **Imputation**: rate/recency features may be NaN (users with no opens,
  no channel, etc.). Imputed with the train median.
- **Global seed**: SEED=42 in every splitter/estimator (config).

Run standalone:
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

from src.features import FEATURES_PATH
from src.logging_utils import (
    MODELS_DIR,
    RUN_TIMESTAMP,
    load_config,
    print_checkpoint,
    setup_logger,
)


# ----------------------------------------------------------------------------
# Helper types
# ----------------------------------------------------------------------------


@dataclass
class SplitData:
    """Container for the train/test split, separated into features/target."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    @property
    def train_prevalence(self) -> float:
        return float(self.y_train.mean())


# ----------------------------------------------------------------------------
# Loading and split
# ----------------------------------------------------------------------------


NON_FEATURE_COLUMNS = {
    "client_id",
    "target_churn_comunicacion",
    "n_msgs_eval",
    "n_opens_eval",
    "n_clicks_eval",
}


def _split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Splits the DataFrame into X (numeric features) and target."""
    y = df["target_churn_comunicacion"].astype("int8")
    feature_columns = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    X = df[feature_columns].copy()
    # Datetime columns (last open/click timestamps) are NOT direct features:
    # they were already transformed into the `dias_desde_ultimo_*` recency
    # features in Phase 2. Dropped explicitly in case they remain in the
    # parquet.
    X = X.select_dtypes(exclude=["datetime64[ns]", "datetime64"])
    return X, y


def prepare_split(df: pd.DataFrame, seed: int, logger) -> SplitData:
    """Stratified 80/20 train/test split."""
    X, y = _split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    logger.info(
        "Split ready: train=%d (prev=%.4f), test=%d (prev=%.4f), features=%d",
        len(X_train),
        float(y_train.mean()),
        len(X_test),
        float(y_test.mean()),
        X_train.shape[1],
    )
    return SplitData(X_train, X_test, y_train, y_test)


# ----------------------------------------------------------------------------
# Pipelines
# ----------------------------------------------------------------------------


def _preproc_steps(standardize: bool) -> list[tuple[str, Any]]:
    """Common preprocessing steps: imputation + (optional) scaling."""
    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if standardize:
        steps.append(("scaler", StandardScaler()))
    return steps


def _pipeline_with_smote(
    preproc_steps: list[tuple[str, Any]],
    estimator: Any,
    apply_smote: bool,
    seed: int,
) -> ImbPipeline:
    """Builds an imblearn Pipeline with optional SMOTE before the estimator.

    SMOTE goes AFTER preprocessing so scaling does not introduce NaNs, and
    BEFORE the estimator. imblearn applies SMOTE only during .fit (not in
    .transform/.predict nor on the validation fold during cross-validation).
    """
    steps = list(preproc_steps)
    if apply_smote:
        steps.append(("smote", SMOTE(random_state=seed)))
    steps.append(("clf", estimator))
    return ImbPipeline(steps)


# ----------------------------------------------------------------------------
# Model definitions + searches
# ----------------------------------------------------------------------------


def build_logreg_pipeline(apply_smote: bool, seed: int) -> ImbPipeline:
    """LR with default regularization (paper: baseline, no search)."""
    steps = _preproc_steps(standardize=True)
    return _pipeline_with_smote(
        steps,
        LogisticRegression(max_iter=1000, random_state=seed),
        apply_smote=apply_smote,
        seed=seed,
    )


def build_rf_pipeline(apply_smote: bool, seed: int) -> ImbPipeline:
    steps = _preproc_steps(standardize=False)
    return _pipeline_with_smote(
        steps,
        RandomForestClassifier(random_state=seed, n_jobs=-1),
        apply_smote=apply_smote,
        seed=seed,
    )


def build_xgb_pipeline(apply_smote: bool, seed: int) -> ImbPipeline:
    steps = _preproc_steps(standardize=False)
    # DECISION: if SMOTE is active, scale_pos_weight=1 (imbalance corrections
    # are never stacked). Otherwise scale_pos_weight is left as a search
    # hyperparameter.
    xgb_kwargs = dict(
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="auc",
    )
    if apply_smote:
        xgb_kwargs["scale_pos_weight"] = 1
    return _pipeline_with_smote(
        steps,
        XGBClassifier(**xgb_kwargs),
        apply_smote=apply_smote,
        seed=seed,
    )


def _rf_grid() -> dict:
    """Bounded hyperparameter space for RF (random search, 20 iters)."""
    return {
        "clf__n_estimators": [200, 400, 600, 800],
        "clf__max_depth": [None, 8, 16, 24],
        "clf__min_samples_split": [2, 5, 10, 20],
        "clf__min_samples_leaf": [1, 2, 4, 8],
        "clf__max_features": ["sqrt", "log2"],
    }


def _xgb_grid(apply_smote: bool) -> dict:
    """Bounded hyperparameter space for XGBoost."""
    grid = {
        "clf__n_estimators": [200, 400, 600, 800],
        "clf__max_depth": [4, 6, 8, 10],
        "clf__learning_rate": [0.03, 0.05, 0.1, 0.2],
        "clf__subsample": [0.7, 0.8, 1.0],
        "clf__colsample_bytree": [0.7, 0.8, 1.0],
        "clf__reg_lambda": [0.5, 1.0, 2.0],
    }
    if not apply_smote:
        # Without SMOTE, do tune scale_pos_weight to address imbalance.
        grid["clf__scale_pos_weight"] = [1.0, 3.0, 5.0, 10.0]
    return grid


def _stratified_subsample(
    X: pd.DataFrame,
    y: pd.Series,
    n_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Target-stratified subsample. If `n_samples >= len(X)`, returns everything."""
    if n_samples >= len(X):
        return X, y
    # train_test_split with stratify=y yields a stratified partition of the
    # requested size (we keep the "train" side and discard the rest).
    X_sub, _, y_sub, _ = train_test_split(
        X, y,
        train_size=n_samples,
        random_state=seed,
        stratify=y,
    )
    return X_sub, y_sub


def run_random_search(
    pipeline: ImbPipeline,
    grid: dict,
    data: SplitData,
    n_iter: int,
    cv_folds: int,
    seed: int,
    logger,
    subsample_size: int | None = None,
) -> RandomizedSearchCV:
    """Random search with stratified CV, optimizing AUC-ROC.

    If `subsample_size` is provided and smaller than the train size, the
    search runs over a stratified subsample. This does NOT affect the
    semantics of the found hyperparameters: it only speeds up the search.
    The refit over the full dataset happens outside this function with
    `_refit_on_full_train`.
    """
    if subsample_size is not None and subsample_size < len(data.X_train):
        X_search, y_search = _stratified_subsample(
            data.X_train, data.y_train, subsample_size, seed
        )
        logger.info(
            "Search subsample: %d (of %d train), prevalence=%.4f",
            len(X_search),
            len(data.X_train),
            float(y_search.mean()),
        )
    else:
        X_search, y_search = data.X_train, data.y_train

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=grid,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,  # estimators already use n_jobs=-1 internally
        random_state=seed,
        # No refit here over X_search: the refit happens outside, over the
        # full train, with the best_params found.
        refit=False,
        return_train_score=False,
        verbose=1,
    )
    t0 = time.time()
    search.fit(X_search, y_search)
    elapsed = time.time() - t0
    logger.info(
        "Random search OK: best_auc=%.4f, params=%s, %.1fs",
        search.best_score_,
        search.best_params_,
        elapsed,
    )
    return search


def _refit_on_full_train(
    pipeline: ImbPipeline,
    best_params: dict,
    data: SplitData,
    logger,
) -> ImbPipeline:
    """Applies `best_params` to the pipeline and fits it on the WHOLE X_train.

    D8 pattern: the search runs over a subsample (fast), the final model
    leverages the full train (~7.9M) to maximize signal.
    """
    pipeline = pipeline.set_params(**best_params)
    t0 = time.time()
    pipeline.fit(data.X_train, data.y_train)
    logger.info(
        "Refit on full train (%d samples) OK in %.1fs",
        len(data.X_train),
        time.time() - t0,
    )
    return pipeline


# ----------------------------------------------------------------------------
# Model persistence
# ----------------------------------------------------------------------------


def _model_path(name: str) -> str:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return str(MODELS_DIR / f"model_{name}_{RUN_TIMESTAMP}.pkl")


def _persist_model(
    name: str,
    estimator: Any,
    data: SplitData,
    smote_applied: bool,
    best_params: dict | None,
    logger,
) -> str:
    """Serializes the trained model + split metadata into a single .pkl."""
    artifact = {
        "name": name,
        "estimator": estimator,
        "feature_names": list(data.X_train.columns),
        "smote_applied": smote_applied,
        "best_params": best_params,
        "timestamp": RUN_TIMESTAMP,
        "n_train": len(data.X_train),
        "n_test": len(data.X_test),
        "train_prevalence": data.train_prevalence,
    }
    path = _model_path(name)
    with open(path, "wb") as f:
        pickle.dump(artifact, f)
    logger.info("Model persisted: %s", path)
    return path


# ----------------------------------------------------------------------------
# Phase 3 orchestrator
# ----------------------------------------------------------------------------


def run_phase_3(force: bool = False) -> dict:
    """Trains the 3 models and returns a dict with artifact paths.

    The `force` parameter does not change the computation (there is no
    caching: every run produces new timestamped pkl files). Accepted for
    consistency with the run.py CLI.
    """
    del force  # currently no effect; see docstring
    logger = setup_logger("phase_3_train")
    t0 = time.time()
    logger.info("=== PHASE 3 START - training ===")

    config = load_config()
    seed = config["seed"]
    smote_threshold = config["smote_threshold"]
    n_iter = config["random_search_n_iter"]
    cv_folds = config["cv_folds"]
    search_subsample = config.get("search_subsample")

    # 1) Load features.
    if not FEATURES_PATH.exists():
        raise RuntimeError(
            f"{FEATURES_PATH} does not exist. Run Phase 2 first (python -m src.features)."
        )
    df = pd.read_parquet(FEATURES_PATH)
    logger.info("Features loaded: %d users, %d columns", *df.shape)

    # 2) Stratified split and SMOTE decision.
    data = prepare_split(df, seed=seed, logger=logger)
    apply_smote = data.train_prevalence < smote_threshold
    logger.info(
        "Train prevalence=%.4f, threshold=%.2f -> SMOTE=%s",
        data.train_prevalence,
        smote_threshold,
        apply_smote,
    )

    artifacts: dict[str, str] = {}

    # 3) Logistic Regression (no search, direct fit).
    logger.info("--- Training Logistic Regression (baseline) ---")
    pipe_lr = build_logreg_pipeline(apply_smote=apply_smote, seed=seed)
    t_lr = time.time()
    pipe_lr.fit(data.X_train, data.y_train)
    logger.info("LR trained in %.1fs", time.time() - t_lr)
    artifacts["logreg"] = _persist_model(
        "logreg", pipe_lr, data, apply_smote, best_params=None, logger=logger
    )

    # 4) Random Forest with random search (on subsample) + refit on full.
    logger.info(
        "--- Training Random Forest (random search %d iters, subsample=%s) ---",
        n_iter, search_subsample,
    )
    pipe_rf = build_rf_pipeline(apply_smote=apply_smote, seed=seed)
    search_rf = run_random_search(
        pipe_rf, _rf_grid(), data, n_iter=n_iter, cv_folds=cv_folds,
        seed=seed, logger=logger, subsample_size=search_subsample,
    )
    pipe_rf_final = _refit_on_full_train(pipe_rf, search_rf.best_params_, data, logger)
    artifacts["rf"] = _persist_model(
        "rf", pipe_rf_final, data, apply_smote,
        best_params=search_rf.best_params_, logger=logger,
    )

    # 5) XGBoost with random search (on subsample) + refit on full.
    logger.info(
        "--- Training XGBoost (random search %d iters, subsample=%s) ---",
        n_iter, search_subsample,
    )
    pipe_xgb = build_xgb_pipeline(apply_smote=apply_smote, seed=seed)
    search_xgb = run_random_search(
        pipe_xgb, _xgb_grid(apply_smote), data,
        n_iter=n_iter, cv_folds=cv_folds, seed=seed, logger=logger,
        subsample_size=search_subsample,
    )
    pipe_xgb_final = _refit_on_full_train(pipe_xgb, search_xgb.best_params_, data, logger)
    artifacts["xgb"] = _persist_model(
        "xgb", pipe_xgb_final, data, apply_smote,
        best_params=search_xgb.best_params_, logger=logger,
    )

    # 6) Persist the split (so evaluate.py uses exactly the same one).
    split_path = MODELS_DIR / f"split_{RUN_TIMESTAMP}.pkl"
    with open(split_path, "wb") as f:
        pickle.dump(
            {
                "X_train": data.X_train,
                "X_test": data.X_test,
                "y_train": data.y_train,
                "y_test": data.y_test,
                "seed": seed,
                "smote_applied": apply_smote,
                "timestamp": RUN_TIMESTAMP,
            },
            f,
        )
    artifacts["split"] = str(split_path)
    logger.info("Split persisted to %s", split_path)

    elapsed = time.time() - t0
    print_checkpoint(
        logger,
        "Phase 3 - Training",
        {
            "Models trained": "logreg, rf, xgb",
            "SMOTE applied": apply_smote,
            "Train size": f"{len(data.X_train):,}",
            "Test size": f"{len(data.X_test):,}",
            "Features": data.X_train.shape[1],
            "Best AUC (CV) RF": f"{search_rf.best_score_:.4f}",
            "Best AUC (CV) XGB": f"{search_xgb.best_score_:.4f}",
            "Artifacts": artifacts,
            "Total time (s)": round(elapsed, 1),
            "Next phase": "Phase 4 - src/evaluate.py",
        },
    )
    return artifacts


if __name__ == "__main__":
    run_phase_3()
