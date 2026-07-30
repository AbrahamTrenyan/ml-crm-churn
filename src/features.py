"""Pipeline Phase 2: per-user feature engineering.

Builds the behavioral feature vector described in section IV.C of the paper,
iterating `messages.csv.gz` in chunks and aggregating metrics per `client_id`
without materializing the full dataset in RAM.

NOTE ON NAMING: the persisted feature/target column names (e.g.
`tasa_open_email`, `dias_antiguedad`, `target_churn_comunicacion`) are kept
in Spanish on purpose: they are embedded in the published figures, the
results report and the trained model artifacts. Renaming them would require
re-running the full pipeline (~12 h) to regenerate every artifact.

General outline:
1. Load config -> get `observation_cutoff` and `evaluation_cutoff`.
2. Pre-load campaigns (lookup by campaign_id) and client_first_purchase_date.
3. Iterate messages.csv.gz in chunks; for each chunk:
   a. Filter to the union of windows (obs + eval).
   b. Flag is_obs / is_eval using the cutoff date.
   c. Join campaign info (is_trigger, has_personalization).
   d. Aggregate counts and maxes by (client_id, channel) and by client_id.
   e. Merge into the global accumulators.
4. Compute derived features (rates, ratios, recencies).
5. Join customer tenure.
6. Build the binary target over the evaluation window.
7. Filter to "evaluable" users (received at least one message in eval).
8. Persist parquet to `outputs/features/features_per_user.parquet`.

DECISION: incremental per-chunk aggregation (no global concat). Each chunk
produces a small DataFrame (~max 460k unique client_id rows) which is
combined with the accumulator via `.add(fill_value=0)` for sums and a merge
followed by `max` for timestamps. This avoids blowing up RAM with 721M rows.

DECISION: the pipeline's "strong" temporal split is features (observation
window) vs target (evaluation window). The train/test split inside Phase 3
is stratified by target -consistent with the paper, section IV.D-, not
temporal. The anti-leakage rule holds because no feature looks at data
after `evaluation_cutoff`.

Run standalone:
    python -m src.features
"""
from __future__ import annotations

import gc
import time

import numpy as np
import pandas as pd

from src.data import iter_messages, load_campaigns, load_first_purchase
from src.logging_utils import (
    FEATURES_DIR,
    load_config,
    print_checkpoint,
    setup_logger,
)

# ----------------------------------------------------------------------------
# Column and channel constants
# ----------------------------------------------------------------------------

# DECISION: only channels with significant volume in the dataset are used
# (paper sec. IV.A). Remaining channels are excluded from modeling.
TARGET_CHANNELS: tuple[str, ...] = ("email", "mobile_push")

# Columns read from messages.csv.gz for Phase 2. Conservative: the minimum
# needed to build every feature in the closed list (decision D3).
PHASE2_MESSAGE_COLUMNS: list[str] = [
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

# Short windows used for the "recent open rate" features (paper sec. IV.C).
RECENT_DAYS_SHORT: int = 30
RECENT_DAYS_LONG: int = 60


# ----------------------------------------------------------------------------
# Setup: temporal cutoffs and auxiliary lookups
# ----------------------------------------------------------------------------


def _load_time_cutoffs() -> dict:
    """Reads `window_dates` from the config and returns the Timestamps."""
    config = load_config()
    wd = config["window_dates"]
    if wd.get("observation_cutoff") is None or wd.get("evaluation_cutoff") is None:
        raise RuntimeError(
            "config.json::window_dates has no valid cutoffs. "
            "Run Phase 1 first (python -m src.data)."
        )
    obs_cutoff = pd.Timestamp(wd["observation_cutoff"])
    eval_cutoff = pd.Timestamp(wd["evaluation_cutoff"])
    max_dt = pd.Timestamp(wd["max_sent_at"])
    return {
        "observation_cutoff": obs_cutoff,
        "evaluation_cutoff": eval_cutoff,
        "max_sent_at": max_dt,
        "recent_cutoff_30d": eval_cutoff - pd.Timedelta(days=RECENT_DAYS_SHORT),
        "recent_cutoff_60d": eval_cutoff - pd.Timedelta(days=RECENT_DAYS_LONG),
    }


def _build_campaigns_lookup() -> pd.DataFrame:
    """Small lookup campaign_id -> (is_trigger, has_personalization).

    Reduced to a DataFrame indexed by `campaign_id` with two boolean
    columns, to do a vectorized `merge` against each messages chunk.
    """
    df = load_campaigns()
    # DECISION: `campaign_type` is categorical -> compare against str and
    # cast to int8. `subject_with_personalization` is read as object with
    # python bools (True/False) and NaN for non-bulk campaigns. We use
    # `fillna(False)` + astype("bool") to neutralize the NaNs and cast to
    # int8 without losing semantics.
    is_trigger = (df["campaign_type"].astype("object") == "trigger").astype("int8")
    has_personalization = (
        df["subject_with_personalization"].fillna(False).astype(bool).astype("int8")
    )
    lookup = pd.DataFrame(
        {
            "is_trigger": is_trigger.to_numpy(),
            "has_personalization": has_personalization.to_numpy(),
        },
        index=df["id"].astype("int64"),
    )
    lookup.index.name = "campaign_id"
    return lookup


# ----------------------------------------------------------------------------
# Aggregation helpers
# ----------------------------------------------------------------------------


# DECISION: sums are stored as `float32` (not float64). Each cell is a
# message/event count with a practical bound of a few thousand, well within
# float32 range. This halves the accumulator memory (16M users x 29 columns
# go from ~3.7 GB to ~1.9 GB) and avoids the swap-thrashing that killed the
# first run over the full dataset.
DTYPE_SUMS: str = "float32"


def _empty_sums_df() -> pd.DataFrame:
    """Skeleton of the sums DataFrame, indexed by client_id."""
    columns = _sum_columns()
    return pd.DataFrame(columns=columns, dtype=DTYPE_SUMS)


def _empty_maxes_df() -> pd.DataFrame:
    """Skeleton of the timestamp-max DataFrame, indexed by client_id."""
    columns = _max_columns()
    return pd.DataFrame(columns=columns, dtype="datetime64[ns]")


def _sum_columns() -> list[str]:
    """List of scalar columns aggregated by sum across chunks."""
    cols = []
    for channel in TARGET_CHANNELS:
        cols += [
            f"n_msgs_{channel}",
            f"n_opens_{channel}",
            f"n_clicks_{channel}",
            f"n_hard_bounces_{channel}",
            f"n_soft_bounces_{channel}",
            f"n_unsubs_{channel}",
            f"n_complaints_{channel}",
            f"n_msgs_30d_{channel}",
            f"n_opens_30d_{channel}",
            f"n_msgs_60d_{channel}",
            f"n_opens_60d_{channel}",
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


def _max_columns() -> list[str]:
    """List of timestamp columns aggregated by max across chunks."""
    cols = []
    for channel in TARGET_CHANNELS:
        cols += [f"ultimo_open_{channel}", f"ultimo_click_{channel}"]
    return cols


# ----------------------------------------------------------------------------
# Per-chunk processing
# ----------------------------------------------------------------------------


def _enrich_chunk(
    chunk: pd.DataFrame,
    cutoffs: dict,
    campaigns_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Filters the chunk to the union of windows and adds support columns.

    Adds:
    - `is_obs`, `is_eval` (temporal masks).
    - `is_trigger`, `has_personalization` (merge with campaigns).
    - `in_30d`, `in_60d` (recency masks within obs).
    """
    # 1) Temporal filter: discard everything before observation_cutoff (D1).
    chunk = chunk.loc[chunk["sent_at"] >= cutoffs["observation_cutoff"]]
    if chunk.empty:
        return chunk

    chunk = chunk.assign(
        is_obs=chunk["sent_at"] < cutoffs["evaluation_cutoff"],
        is_eval=chunk["sent_at"] >= cutoffs["evaluation_cutoff"],
    )
    # Recency masks (only apply within obs).
    chunk["in_30d"] = chunk["is_obs"] & (chunk["sent_at"] >= cutoffs["recent_cutoff_30d"])
    chunk["in_60d"] = chunk["is_obs"] & (chunk["sent_at"] >= cutoffs["recent_cutoff_60d"])

    # 2) Vectorized merge with campaigns (left join; no match -> 0).
    chunk = chunk.merge(
        campaigns_lookup, how="left", left_on="campaign_id", right_index=True
    )
    chunk["is_trigger"] = chunk["is_trigger"].fillna(0).astype("int8")
    chunk["has_personalization"] = (
        chunk["has_personalization"].fillna(0).astype("int8")
    )
    return chunk


def _aggregate_chunk_sums(chunk: pd.DataFrame) -> pd.DataFrame:
    """Sums every 'count'-type feature by client_id for the chunk.

    Returns a DataFrame indexed by client_id with the `_sum_columns` columns.
    """
    df_obs = chunk.loc[chunk["is_obs"]]
    df_eval = chunk.loc[chunk["is_eval"]]
    out_pieces: list[pd.DataFrame] = []

    # Per-channel sums (target channels only).
    for channel in TARGET_CHANNELS:
        sub = df_obs.loc[df_obs["channel"] == channel]
        if sub.empty:
            continue
        g = sub.groupby("client_id", observed=True)
        piece = pd.DataFrame(
            {
                f"n_msgs_{channel}": g.size(),
                f"n_opens_{channel}": g["is_opened"].sum(),
                f"n_clicks_{channel}": g["is_clicked"].sum(),
                f"n_hard_bounces_{channel}": g["is_hard_bounced"].sum(),
                f"n_soft_bounces_{channel}": g["is_soft_bounced"].sum(),
                f"n_unsubs_{channel}": g["is_unsubscribed"].sum(),
                f"n_complaints_{channel}": g["is_complained"].sum(),
                f"n_msgs_30d_{channel}": sub.loc[sub["in_30d"]].groupby("client_id", observed=True).size(),
                f"n_opens_30d_{channel}": sub.loc[sub["in_30d"]].groupby("client_id", observed=True)["is_opened"].sum(),
                f"n_msgs_60d_{channel}": sub.loc[sub["in_60d"]].groupby("client_id", observed=True).size(),
                f"n_opens_60d_{channel}": sub.loc[sub["in_60d"]].groupby("client_id", observed=True)["is_opened"].sum(),
            }
        )
        out_pieces.append(piece)

    # Global sums over OBS (not segmented by channel).
    if not df_obs.empty:
        g_obs = df_obs.groupby("client_id", observed=True)
        out_pieces.append(
            pd.DataFrame(
                {
                    "n_compras_atribuidas": g_obs["is_purchased"].sum(),
                    "n_msgs_obs_total": g_obs.size(),
                    "n_msgs_trigger_obs": g_obs["is_trigger"].sum(),
                    "n_msgs_personalizados_obs": g_obs["has_personalization"].sum(),
                }
            )
        )

    # Sums over EVAL (target and support).
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

    # Join by client_id (outer): each user appears exactly once.
    joined = pd.concat(out_pieces, axis=1).fillna(0)
    # Cast to float32 to halve the accumulator's memory pressure.
    return joined.astype(DTYPE_SUMS)


def _aggregate_chunk_maxes(chunk: pd.DataFrame) -> pd.DataFrame:
    """Computes open/click timestamp maxes by (client_id, channel) within obs."""
    df_obs = chunk.loc[chunk["is_obs"]]
    cols: dict[str, pd.Series] = {}
    for channel in TARGET_CHANNELS:
        sub = df_obs.loc[df_obs["channel"] == channel]
        if sub.empty:
            continue
        g = sub.groupby("client_id", observed=True)
        # DECISION: we use `opened_first_time_at` (NaT if never opened) and
        # take the max over all of the user's messages in obs. Equivalent to
        # the user's last open on that channel within obs.
        cols[f"ultimo_open_{channel}"] = g["opened_first_time_at"].max()
        cols[f"ultimo_click_{channel}"] = g["clicked_first_time_at"].max()
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols)


# DECISION: the accumulator (DataFrame indexed by client_id) grows to ~16M
# rows. Combining it with every chunk via `.add(fill_value=0)` or
# `pd.concat+groupby.max` is O(accumulator size) per chunk -> quadratic in
# total. Refactor: chunk-level DataFrames are accumulated in a list
# (`pending`) and only compacted every `CHUNKS_PER_COMPACTION` chunks. This
# turns 1442 large merges into ~70, keeping the result mathematically
# identical.
#
# Value: 20 (lowered from 50 after the first real run showed that with 50
# pending chunks the RAM peak during compaction crosses the swap threshold
# on a 16 GB machine).
CHUNKS_PER_COMPACTION: int = 20


def _compact_sums(sums_acc: pd.DataFrame, pending: list[pd.DataFrame]) -> pd.DataFrame:
    """Combines the sums accumulator with N pending DataFrames in one groupby."""
    non_empty = [df for df in pending if not df.empty]
    if not non_empty:
        return sums_acc
    combined = pd.concat([sums_acc] + non_empty)
    if combined.empty:
        return sums_acc
    # `groupby(level=0).sum()` preserves the input's float32 dtype.
    return combined.groupby(level=0).sum()


def _compact_maxes(maxes_acc: pd.DataFrame, pending: list[pd.DataFrame]) -> pd.DataFrame:
    """Combines the timestamp accumulator with N pending DataFrames in one groupby."""
    non_empty = [df for df in pending if not df.empty]
    if not non_empty:
        return maxes_acc
    combined = pd.concat([maxes_acc] + non_empty)
    if combined.empty:
        return maxes_acc
    return combined.groupby(level=0).max()


# ----------------------------------------------------------------------------
# Full-scan orchestration
# ----------------------------------------------------------------------------


def build_accumulators(logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full scan of messages.csv.gz, returning (sums, maxes) per user.

    Uses batch compaction (every `CHUNKS_PER_COMPACTION` chunks) to amortize
    the cost of the large merges and keep iteration time approximately
    linear instead of quadratic.
    """
    cutoffs = _load_time_cutoffs()
    lookup = _build_campaigns_lookup()
    logger.info(
        "Time cutoffs: obs=%s, eval=%s, max=%s",
        cutoffs["observation_cutoff"],
        cutoffs["evaluation_cutoff"],
        cutoffs["max_sent_at"],
    )
    logger.info(
        "Batch compaction every %d chunks (accumulators merged per batch).",
        CHUNKS_PER_COMPACTION,
    )

    sums_acc = _empty_sums_df()
    maxes_acc = _empty_maxes_df()
    pending_sums: list[pd.DataFrame] = []
    pending_maxes: list[pd.DataFrame] = []
    n_chunks = 0
    n_rows_processed = 0
    t0 = time.time()

    for chunk in iter_messages(columns=PHASE2_MESSAGE_COLUMNS):
        n_chunks += 1
        chunk = _enrich_chunk(chunk, cutoffs, lookup)
        if chunk.empty:
            if n_chunks % CHUNKS_PER_COMPACTION == 0:
                sums_acc = _compact_sums(sums_acc, pending_sums)
                maxes_acc = _compact_maxes(maxes_acc, pending_maxes)
                pending_sums, pending_maxes = [], []
            continue
        n_rows_processed += len(chunk)

        pending_sums.append(_aggregate_chunk_sums(chunk))
        pending_maxes.append(_aggregate_chunk_maxes(chunk))

        if n_chunks % CHUNKS_PER_COMPACTION == 0:
            t_compact = time.time()
            sums_acc = _compact_sums(sums_acc, pending_sums)
            maxes_acc = _compact_maxes(maxes_acc, pending_maxes)
            # Release references before gc so the pending chunks' buffers
            # are actually reclaimed.
            pending_sums.clear()
            pending_maxes.clear()
            gc.collect()
            elapsed = time.time() - t0
            logger.info(
                "  ... %d chunks, %d accumulated users, %.1fs (compaction %.1fs)",
                n_chunks,
                len(sums_acc),
                elapsed,
                time.time() - t_compact,
            )

    # Final compaction in case anything is still pending.
    if pending_sums or pending_maxes:
        sums_acc = _compact_sums(sums_acc, pending_sums)
        maxes_acc = _compact_maxes(maxes_acc, pending_maxes)

    logger.info(
        "Scan finished: %d chunks, %d rows within windows, %d users.",
        n_chunks,
        n_rows_processed,
        len(sums_acc),
    )
    return sums_acc, maxes_acc


# ----------------------------------------------------------------------------
# Derived features
# ----------------------------------------------------------------------------


def _compute_rates_and_recencies(
    df: pd.DataFrame, cutoffs: dict
) -> pd.DataFrame:
    """Adds rates (open/click), recent rates and days-since-last-action."""
    eval_cutoff = cutoffs["evaluation_cutoff"]

    for channel in TARGET_CHANNELS:
        # Rates (NaN if the user received no messages on the channel/window).
        df[f"tasa_open_{channel}"] = df[f"n_opens_{channel}"] / df[f"n_msgs_{channel}"].replace(0, np.nan)
        df[f"tasa_click_{channel}"] = df[f"n_clicks_{channel}"] / df[f"n_msgs_{channel}"].replace(0, np.nan)
        df[f"tasa_open_30d_{channel}"] = (
            df[f"n_opens_30d_{channel}"] / df[f"n_msgs_30d_{channel}"].replace(0, np.nan)
        )
        df[f"tasa_open_60d_{channel}"] = (
            df[f"n_opens_60d_{channel}"] / df[f"n_msgs_60d_{channel}"].replace(0, np.nan)
        )
        # Recencies: days from the last open/click to the close of obs.
        # NaN if the user never opened/clicked on that channel.
        df[f"dias_desde_ultimo_open_{channel}"] = (
            (eval_cutoff - df[f"ultimo_open_{channel}"]).dt.total_seconds() / 86400.0
        )
        df[f"dias_desde_ultimo_click_{channel}"] = (
            (eval_cutoff - df[f"ultimo_click_{channel}"]).dt.total_seconds() / 86400.0
        )
    return df


def _compute_cross_channel_features(df: pd.DataFrame, cutoffs: dict) -> pd.DataFrame:
    """Cross-channel features: channels with opens, rate diff, recent ratio."""
    # n_canales_con_open: how many target channels have >=1 open.
    flags = pd.concat(
        [(df[f"n_opens_{c}"] > 0).astype("int8") for c in TARGET_CHANNELS], axis=1
    )
    df["n_canales_con_open"] = flags.sum(axis=1).astype("int8")

    # diff_tasa_open: difference between per-channel open rates (paper).
    # Reported as email - mobile_push; NaN if either channel has no messages.
    df["diff_tasa_open_email_vs_push"] = (
        df["tasa_open_email"] - df["tasa_open_mobile_push"]
    )

    # ratio_actividad_30d_vs_historico: opens in the last 30 days vs the
    # expected historical average under a uniform distribution within obs.
    obs_days = (
        cutoffs["evaluation_cutoff"] - cutoffs["observation_cutoff"]
    ).total_seconds() / 86400.0
    opens_30d_total = sum(df[f"n_opens_30d_{c}"] for c in TARGET_CHANNELS)
    opens_obs_total = sum(df[f"n_opens_{c}"] for c in TARGET_CHANNELS)
    expected_30d = opens_obs_total * (RECENT_DAYS_SHORT / obs_days)
    df["ratio_actividad_30d_vs_historico"] = opens_30d_total / expected_30d.replace(
        0, np.nan
    )
    return df


def _compute_campaign_features(df: pd.DataFrame) -> pd.DataFrame:
    """Proportions of trigger and personalized messages (paper sec. IV.C)."""
    total = df["n_msgs_obs_total"].replace(0, np.nan)
    df["prop_trigger"] = df["n_msgs_trigger_obs"] / total
    df["prop_personalizacion"] = df["n_msgs_personalizados_obs"] / total
    return df


def _add_tenure(df: pd.DataFrame, cutoffs: dict) -> pd.DataFrame:
    """Joins `client_first_purchase_date` and computes tenure in days."""
    first_purchase = load_first_purchase().set_index("client_id")
    df = df.merge(
        first_purchase, how="left", left_index=True, right_index=True
    )
    df["dias_antiguedad"] = (
        (cutoffs["observation_cutoff"] - df["first_purchase_date"]).dt.total_seconds()
        / 86400.0
    )
    df = df.drop(columns=["first_purchase_date"])
    return df


def _build_target(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Defines the binary target and filters to evaluable users.

    target_churn_comunicacion = 1 if the user received >=1 message in the
    evaluation window but opened and clicked none; 0 otherwise. Users with
    no messages in eval are discarded (not evaluable).
    """
    n_before = len(df)
    df["n_msgs_eval"] = df["n_msgs_eval"].fillna(0)
    df["n_opens_eval"] = df["n_opens_eval"].fillna(0)
    df["n_clicks_eval"] = df["n_clicks_eval"].fillna(0)

    evaluable = df["n_msgs_eval"] > 0
    df = df.loc[evaluable].copy()
    n_filtered = n_before - len(df)
    logger.info(
        "Filtered %d users with no messages in eval. Evaluable: %d",
        n_filtered,
        len(df),
    )

    df["target_churn_comunicacion"] = (
        (df["n_opens_eval"] == 0) & (df["n_clicks_eval"] == 0)
    ).astype("int8")
    return df


# ----------------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------------


FEATURES_PATH = FEATURES_DIR / "features_per_user.parquet"


def _persist(df: pd.DataFrame, logger) -> None:
    """Writes the final DataFrame to parquet (pyarrow)."""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    df = df.reset_index().rename(columns={"index": "client_id"})
    df["client_id"] = df["client_id"].astype("int64")
    df.to_parquet(FEATURES_PATH, engine="pyarrow", index=False)
    logger.info("Features persisted to %s", FEATURES_PATH)


# ----------------------------------------------------------------------------
# Phase 2 orchestrator
# ----------------------------------------------------------------------------


def run_phase_2(force: bool = False) -> None:
    """Phase 2 entry point. Builds and persists per-user features.

    If the parquet already exists and `force=True` is not passed, it is not
    recomputed.
    """
    logger = setup_logger("phase_2_features")
    t0 = time.time()
    logger.info("=== PHASE 2 START - feature engineering ===")

    if FEATURES_PATH.exists() and not force:
        logger.info("Parquet already exists (%s). Use force=True to recompute.", FEATURES_PATH)
        return

    cutoffs = _load_time_cutoffs()

    # 1) Scan + aggregation.
    sums_acc, maxes_acc = build_accumulators(logger)

    # 2) Join sums + maxes into a single per-user DataFrame.
    df = sums_acc.join(maxes_acc, how="outer")
    df.index.name = "client_id"
    # Sum columns left NaN after the outer join are real zeros.
    for col in _sum_columns():
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 3) Derived features and final joins.
    df = _compute_rates_and_recencies(df, cutoffs)
    df = _compute_cross_channel_features(df, cutoffs)
    df = _compute_campaign_features(df)
    df = _add_tenure(df, cutoffs)
    df = _build_target(df, logger)

    # Drop intermediate datetime columns: already replaced by the
    # `dias_desde_*` features and not useful as direct model inputs.
    dt_columns = [c for c in _max_columns() if c in df.columns]
    if dt_columns:
        df = df.drop(columns=dt_columns)

    # 4) Guardrail: target prevalence.
    # DECISION (real run over the full dataset, 2026-05-19): the observed
    # prevalence is ~57%, not the "minority" anticipated by the paper. The
    # target definition matches paper sec. IV.B exactly (eval >=1 msg, 0
    # opens, 0 clicks), so the deviation is a property of the REES46 dataset
    # for the 2022-10-23 -> 2023-04-23 window, not a bug. A guardrail is
    # kept for pathological edges (<=1% or >=80%) and the 50-80% range is
    # logged as a methodological warning (a majority positive class means F1
    # on class 1 stops being the natural objective; class 0 is also
    # reported).
    prevalence = float(df["target_churn_comunicacion"].mean())
    if prevalence < 0.01 or prevalence > 0.8:
        logger.error(
            "Churn prevalence out of range: %.4f. Review the target definition.",
            prevalence,
        )
        raise RuntimeError(
            f"Prevalence {prevalence:.4f} outside the acceptable range (1%-80%)."
        )
    if prevalence > 0.5:
        logger.warning(
            "Churn prevalence = %.4f (>50%%). Positive class is the majority. "
            "Documented as decision D7. Pipeline continues.",
            prevalence,
        )

    # 5) Persist.
    _persist(df, logger)

    elapsed = time.time() - t0
    print_checkpoint(
        logger,
        "Phase 2 - Feature engineering",
        {
            "Evaluable users": f"{len(df):,}",
            "Features per user": df.shape[1] - 1,
            "Target prevalence (churn)": f"{prevalence:.4f}",
            "Parquet generated": FEATURES_PATH,
            "Total time (s)": round(elapsed, 1),
            "Next phase": "Phase 3 - src/train.py",
        },
    )


if __name__ == "__main__":
    run_phase_2()
