"""Pipeline Phase 1: CSV loading and validation.

Responsibilities:
- Verify the integrity of `messages.csv.gz` (full dataset, 24 months) via MD5.
- Load the 4 CSVs with explicit dtypes and 't'/'f' boolean conversion.
- Compute the time windows (decision D1): 18 months of observation and
  6 months of evaluation, both counted backwards from `max(sent_at)`.
- Persist the computed dates in `config.json::window_dates`.
- Emit a checkpoint with rows, unique users and time ranges.

DECISION (D5): the pipeline works with `messages.csv.gz` (21.5 GB
compressed, 24 months, ~721M rows) instead of `messages-demo.csv` (10M rows,
46 days). The file is ALWAYS processed from gzip via streaming/chunks; it is
never decompressed to disk because the plain file would take ~150 GB while
only ~70 GB are free.

Run standalone:
    python -m src.data
"""
from __future__ import annotations

import hashlib
import time
from typing import Iterator

import pandas as pd

from src.logging_utils import (
    DATA_DIR,
    load_config,
    print_checkpoint,
    save_config,
    setup_logger,
)

# ----------------------------------------------------------------------------
# messages.csv.gz schema
# ----------------------------------------------------------------------------

# DECISION: `client_id` is stored as int64. Real IDs in the dataset are
# ~1.5e18, far above the int32 maximum (~2.1e9). The explicit typing rule is
# kept, but the dtype is bumped from int32 to int64 given the observed range.
MESSAGES_DTYPES: dict[str, str] = {
    "id": "int64",
    "message_id": "string",
    "campaign_id": "Int32",  # nullable: transactional records have no campaign
    "message_type": "category",
    "client_id": "int64",
    "channel": "category",
    "category": "category",
    "platform": "category",
    "email_provider": "category",
    "stream": "category",
}

# The 8 boolean columns come as 't'/'f' in the CSV. They are converted to
# int8 in per-chunk post-processing (faster than a converter in read_csv).
BOOLEAN_COLUMNS: list[str] = [
    "is_opened",
    "is_clicked",
    "is_unsubscribed",
    "is_hard_bounced",
    "is_soft_bounced",
    "is_complained",
    "is_blocked",
    "is_purchased",
]

# All timestamp-like columns in the messages file.
MESSAGES_DATE_COLUMNS: list[str] = [
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

# Column subset used by Phase 1 to validate integrity and compute the time
# windows. Only these 3 are read to avoid loading 32 columns x 721M rows
# into memory.
MINIMAL_VALIDATION_COLUMNS: list[str] = ["client_id", "channel", "sent_at"]

MESSAGES_FILENAME = "messages.csv.gz"

# DECISION: the defensive threshold is set to 400M rows (the paper reports
# ~721M raw, but we want a conservative guardrail that only fires if the
# download was truncated). If the full dataset download is partial, Phase 1
# aborts and warns.
EXPECTED_MESSAGES_ROWS = 400_000_000


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _convert_tf_booleans(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Converts 't'/'f' columns to int8 (1/0). NaN is treated as 0."""
    for col in columns:
        if col in df.columns:
            # Vectorized comparison: any value other than 't' (incl. NaN) -> 0.
            df[col] = (df[col] == "t").astype("int8")
    return df


def _filter_dtypes(columns: list[str] | None) -> dict[str, str]:
    """Returns the subset of MESSAGES_DTYPES applicable to the requested columns."""
    if columns is None:
        return dict(MESSAGES_DTYPES)
    return {k: v for k, v in MESSAGES_DTYPES.items() if k in columns}


def _filter_parse_dates(columns: list[str] | None) -> list[str]:
    """Returns only the date columns present in `columns`."""
    if columns is None:
        return list(MESSAGES_DATE_COLUMNS)
    return [c for c in MESSAGES_DATE_COLUMNS if c in columns]


def _messages_read_kwargs(columns: list[str] | None) -> dict:
    """Builds the common kwargs for `pd.read_csv` over `messages.csv.gz`."""
    path = DATA_DIR / MESSAGES_FILENAME
    dtype_map = _filter_dtypes(columns)
    dates = _filter_parse_dates(columns)
    # DECISION: `compression='gzip'` is passed explicitly to avoid relying on
    # extension-based inference (it also documents the reader).
    return {
        "filepath_or_buffer": path,
        "usecols": columns,
        "dtype": dtype_map,
        "parse_dates": dates if dates else None,
        "compression": "gzip",
    }


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------


def load_messages(
    columns: list[str] | None = None,
    chunksize: int | None = None,
) -> pd.DataFrame:
    """Loads `messages.csv.gz` with dtypes/dates/booleans resolved.

    Args:
        columns: column subset to read (`usecols`). For small subsets
            (3-4 columns) this remains viable. To read the full 32 columns
            use `iter_messages()`.
        chunksize: if given, reads in chunks and concatenates at the end.
            Meant for subsets that do fit in RAM once concatenated.

    Returns:
        DataFrame with the requested columns, dates as datetime64 and
        booleans as int8.

    Raises:
        RuntimeError: if all columns are requested into a single DataFrame.
            The full decompressed dataset is around 100-150 GB and does not
            fit in RAM. Use `iter_messages()` for that case.
    """
    if columns is None:
        raise RuntimeError(
            "load_messages() does not support reading all columns into memory "
            "because `messages.csv.gz` decompresses to ~100-150 GB. "
            "Use `iter_messages(columns=[...])` and aggregate per chunk."
        )

    kwargs = _messages_read_kwargs(columns)
    bools = [c for c in BOOLEAN_COLUMNS if c in columns]

    if chunksize is None:
        df = pd.read_csv(**kwargs)
        return _convert_tf_booleans(df, bools)

    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(chunksize=chunksize, **kwargs):
        parts.append(_convert_tf_booleans(chunk, bools))
    return pd.concat(parts, ignore_index=True)


def iter_messages(
    columns: list[str],
    chunksize: int | None = None,
) -> Iterator[pd.DataFrame]:
    """Iterates `messages.csv.gz` in chunks, with dtypes/dates/booleans resolved.

    Meant for Phase 1 (aggregating global metrics without accumulating the
    whole DataFrame) and Phase 2 (incremental aggregation by `client_id`).

    Args:
        columns: column subset to read. Required so the consumer controls
            the per-chunk DataFrame width.
        chunksize: rows per chunk. If `None`, `chunksize_messages` from
            `config.json` is used.

    Yields:
        DataFrames of at most `chunksize` rows, with dtypes applied and
        booleans converted to int8.
    """
    if not columns:
        raise ValueError("iter_messages() requires a non-empty `columns`.")

    if chunksize is None:
        chunksize = int(load_config().get("chunksize_messages", 500_000))

    kwargs = _messages_read_kwargs(columns)
    bools = [c for c in BOOLEAN_COLUMNS if c in columns]
    for chunk in pd.read_csv(chunksize=chunksize, **kwargs):
        yield _convert_tf_booleans(chunk, bools)


def load_campaigns() -> pd.DataFrame:
    """Loads `campaigns.csv` (~1.9k rows, no memory pressure)."""
    return pd.read_csv(
        DATA_DIR / "campaigns.csv",
        parse_dates=["started_at", "finished_at"],
        dtype={
            "id": "int32",
            "campaign_type": "category",
            "channel": "category",
            "topic": "category",
        },
    )


def load_first_purchase() -> pd.DataFrame:
    """Loads `client_first_purchase_date.csv` (~1.85M rows).

    Covers the range 2021-12-15 -> 2023-12-14, consistent with the full
    dataset window. Used for the tenure feature in Phase 2.
    """
    return pd.read_csv(
        DATA_DIR / "client_first_purchase_date.csv",
        parse_dates=["first_purchase_date"],
        dtype={"client_id": "int64"},
    )


def load_holidays() -> pd.DataFrame:
    """Loads `holidays.csv` (~47 rows, commercial calendar)."""
    return pd.read_csv(
        DATA_DIR / "holidays.csv",
        parse_dates=["date"],
        dtype={"holiday": "category"},
    )


# ----------------------------------------------------------------------------
# MD5 verification
# ----------------------------------------------------------------------------


def verify_messages_md5(logger) -> str:
    """Computes the MD5 of `messages.csv.gz` and compares it with `config.json`.

    If the config has a recorded MD5 and it does not match, a warning is
    emitted but the run is not aborted: it may be a legitimately different
    version of the dataset.
    """
    config = load_config()
    expected_md5 = config.get("dataset_md5", {}).get(MESSAGES_FILENAME)

    path = DATA_DIR / MESSAGES_FILENAME
    hasher = hashlib.md5()
    # 8 MB block reads: balance between throughput and RAM. Over 21.5 GB
    # this takes ~30-60 s on SSD.
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    computed_md5 = hasher.hexdigest()

    if expected_md5 is None:
        logger.info("MD5 computed (no previous reference): %s", computed_md5)
    elif computed_md5 == expected_md5:
        logger.info("MD5 verified OK: %s", computed_md5)
    else:
        logger.warning(
            "MD5 does NOT match. Expected=%s | Computed=%s. "
            "The dataset may have changed; review before continuing.",
            expected_md5,
            computed_md5,
        )
    return computed_md5


# ----------------------------------------------------------------------------
# Chunked aggregation for Phase 1
# ----------------------------------------------------------------------------


def collect_global_metrics(logger) -> dict:
    """Iterates all of `messages.csv.gz` (3 columns) and aggregates global metrics.

    This function replaces the previous "load everything into a DataFrame
    and compute over it" pattern. On the full dataset that would blow up
    RAM. Here we accumulate: min/max of `sent_at`, set of unique
    `client_id`, running sum of `channel.value_counts()` and total row count.

    Returns:
        Dict with n_rows, n_users, channels (dict channel -> count),
        min_sent_at, max_sent_at.
    """
    n_rows = 0
    n_chunks = 0
    unique_clients: set[int] = set()
    channels_acc = pd.Series(dtype="int64")
    min_dt: pd.Timestamp | None = None
    max_dt: pd.Timestamp | None = None

    t_iter = time.time()
    for chunk in iter_messages(columns=MINIMAL_VALIDATION_COLUMNS):
        n_chunks += 1
        n_rows += len(chunk)
        unique_clients.update(chunk["client_id"].unique().tolist())

        chunk_counts = chunk["channel"].value_counts()
        channels_acc = channels_acc.add(chunk_counts, fill_value=0)

        chunk_min = chunk["sent_at"].min()
        chunk_max = chunk["sent_at"].max()
        if pd.notna(chunk_min) and (min_dt is None or chunk_min < min_dt):
            min_dt = chunk_min
        if pd.notna(chunk_max) and (max_dt is None or chunk_max > max_dt):
            max_dt = chunk_max

        # Log every 50 chunks (~25M rows with chunksize=500k) to avoid spam.
        if n_chunks % 50 == 0:
            elapsed = time.time() - t_iter
            rate = n_rows / max(elapsed, 1e-6) / 1_000_000
            logger.info(
                "  ... %d chunks, %d accumulated rows (%.1f M rows/s)",
                n_chunks,
                n_rows,
                rate,
            )

    channels_dict = {str(k): int(v) for k, v in channels_acc.astype("int64").items()}
    return {
        "n_rows": n_rows,
        "n_users": len(unique_clients),
        "channels": channels_dict,
        "min_sent_at": min_dt,
        "max_sent_at": max_dt,
    }


# ----------------------------------------------------------------------------
# Time windows (D1)
# ----------------------------------------------------------------------------


def compute_time_windows(
    min_dt: pd.Timestamp, max_dt: pd.Timestamp, logger
) -> dict:
    """Computes `observation_cutoff` and `evaluation_cutoff` following D1.

    Rule: count backwards from `max(sent_at)`. The last 6 months are the
    evaluation window; the 18 months before it, the observation window.
    Everything prior to those 24 months is discarded in Phase 2.
    """
    # DECISION: fixed windows counted backwards from max(sent_at). 6m for
    # evaluation, the previous 18m for observation. Anything earlier is
    # discarded (D1). Guarantees exact windows consistent with the prediction
    # horizon declared in the paper (sec. IV.B).
    config = load_config()
    obs_months = config["observation_window_months"]
    eval_months = config["evaluation_window_months"]

    eval_cutoff: pd.Timestamp = max_dt - pd.DateOffset(months=eval_months)
    obs_cutoff: pd.Timestamp = eval_cutoff - pd.DateOffset(months=obs_months)

    logger.info("Dataset time range: %s -> %s", min_dt, max_dt)
    logger.info("Observation cutoff (window start): %s", obs_cutoff)
    logger.info("Evaluation cutoff  (window start): %s", eval_cutoff)

    return {
        "min_sent_at": min_dt,
        "max_sent_at": max_dt,
        "observation_cutoff": obs_cutoff,
        "evaluation_cutoff": eval_cutoff,
    }


def persist_windows_in_config(windows: dict, current_md5: str, logger) -> None:
    """Writes the dates and the effective MD5 into `config.json`.

    Removes the old `messages-demo.csv` entry from the `dataset_md5` block
    and leaves a single canonical `messages.csv.gz` key.
    """
    config = load_config()
    config["window_dates"] = {
        "_comment": (
            "Computed in Phase 1 from min/max of sent_at over the full "
            "dataset (messages.csv.gz). See decisions D1 and D5."
        ),
        "min_sent_at": windows["min_sent_at"].isoformat(),
        "max_sent_at": windows["max_sent_at"].isoformat(),
        "observation_cutoff": windows["observation_cutoff"].isoformat(),
        "evaluation_cutoff": windows["evaluation_cutoff"].isoformat(),
    }
    md5_block = config.setdefault("dataset_md5", {})
    md5_block.pop("messages-demo.csv", None)
    md5_block[MESSAGES_FILENAME] = current_md5
    save_config(config)
    logger.info("Dates and MD5 persisted in config.json")


# ----------------------------------------------------------------------------
# Validations
# ----------------------------------------------------------------------------


def validate_metrics(metrics: dict, logger) -> None:
    """Defensive guardrail over the global dataset metrics.

    Aborts if the row count is suspiciously low (rule: "if the loaded
    dataset has fewer rows than expected, stop...").
    """
    n_rows = metrics["n_rows"]
    if n_rows < EXPECTED_MESSAGES_ROWS:
        logger.error(
            "Unexpectedly low row count: %d (expected >=%d). "
            "The download may be truncated.",
            n_rows,
            EXPECTED_MESSAGES_ROWS,
        )
        raise RuntimeError(
            f"messages.csv.gz has {n_rows} rows, far below the "
            f"{EXPECTED_MESSAGES_ROWS} threshold."
        )


# ----------------------------------------------------------------------------
# Phase 1 orchestrator
# ----------------------------------------------------------------------------


def run_phase_1() -> None:
    """Phase 1 entry point. Validate, compute windows and persist."""
    logger = setup_logger("phase_1_data")
    t0 = time.time()
    logger.info("=== PHASE 1 START - loading and validation ===")

    # 1) MD5 verification (not fatal if it differs; warning only).
    current_md5 = verify_messages_md5(logger)

    # 2) Chunked pass over messages.csv.gz: only 3 columns to validate
    #    integrity and compute the time range. Metrics are accumulated
    #    without materializing the full DataFrame (RAM forbids it at this
    #    scale).
    logger.info(
        "Iterating messages.csv.gz in chunks (columns: %s)...",
        MINIMAL_VALIDATION_COLUMNS,
    )
    t_msg = time.time()
    metrics = collect_global_metrics(logger)
    logger.info(
        "Full iteration in %.1fs (%d rows)",
        time.time() - t_msg,
        metrics["n_rows"],
    )

    validate_metrics(metrics, logger)

    # 3) Time windows (D1) + persistence in config.json.
    windows = compute_time_windows(
        metrics["min_sent_at"], metrics["max_sent_at"], logger
    )
    persist_windows_in_config(windows, current_md5, logger)

    # 4) Load + quick validation of the 3 auxiliary files.
    df_campaigns = load_campaigns()
    df_first_purchase = load_first_purchase()
    df_holidays = load_holidays()
    logger.info(
        "Auxiliary files loaded: campaigns=%d, first_purchase=%d, holidays=%d",
        len(df_campaigns),
        len(df_first_purchase),
        len(df_holidays),
    )

    # 5) Final checkpoint.
    elapsed = time.time() - t0
    print_checkpoint(
        logger,
        "Phase 1 - Loading and validation",
        {
            "Rows in messages.csv.gz": f"{metrics['n_rows']:,}",
            "Unique users (client_id)": f"{metrics['n_users']:,}",
            "Channel distribution": metrics["channels"],
            "Rows in campaigns": len(df_campaigns),
            "Rows in client_first_purchase_date": f"{len(df_first_purchase):,}",
            "Rows in holidays": len(df_holidays),
            "Dataset time range": f"{windows['min_sent_at']} -> {windows['max_sent_at']}",
            "Observation cutoff": windows["observation_cutoff"],
            "Evaluation cutoff": windows["evaluation_cutoff"],
            "MD5 messages.csv.gz": current_md5,
            "Total time (s)": round(elapsed, 1),
            "Next phase": "Phase 2 - src/features.py",
        },
    )


if __name__ == "__main__":
    run_phase_1()
