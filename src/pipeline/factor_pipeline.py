"""
Phase 2 Factor Pipeline — Main Orchestrator.

Entry point:  generate_factor_scores(date) → pd.DataFrame

Steps:
  1. Compute raw Mean Reversion signal
  2. Compute raw Sentiment Velocity signal
  3. Compute raw P/B Rank signal
  4. Merge all signals on ticker
  5. Winsorize each raw signal
  6. Z-score normalize each signal cross-sectionally
  7. Validate output schema
  8. Persist to parquet

Output schema:
    date     : datetime64[ns]
    ticker   : str
    mr_z     : float   (Mean Reversion Z-score)
    news_z   : float   (Sentiment Velocity Z-score)
    pb_z     : float   (P/B Value Z-score, inverted so low PB → positive)
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from src.factors.mean_reversion.mean_reversion_signal import compute_mean_reversion_signal
from src.factors.sentiment_velocity.sentiment_velocity import compute_sentiment_velocity_signal
from src.factors.valuation.value_score import compute_pb_signal
from src.normalization.winsorize import winsorize_dataframe
from src.normalization.zscore import zscore_dataframe
from src.pipeline.schemas import validate_factor_output
from src.utils.config import get_settings
from src.utils.logger import get_logger, LatencyTimer

logger = get_logger(__name__)

# ── S&P 500 sample universe (replace with live universe loader in production) ──
SP500_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    "BRK-B", "UNH", "LLY", "JPM", "XOM", "V", "AVGO", "PG",
    "MA", "HD", "CVX", "MRK", "ABBV", "AMD", "COST", "ADBE",
    "CRM", "ACN", "NFLX", "PEP", "KO", "TMO", "CSCO",
    "MCD", "WMT", "BAC", "DIS", "PM", "INTC", "ABT", "WFC",
    "INTU", "QCOM", "IBM", "GE", "CAT", "BA", "GS", "MS",
    "RTX", "HON", "AMGN", "SBUX",
]


def _load_universe() -> list[str]:
    """
    In production, fetch the current S&P 500 constituent list.
    Apply liquidity filter: avg_volume >= min_avg_volume.
    For now, returns the sample universe above.
    """
    return SP500_UNIVERSE


# ── Data quality checks ────────────────────────────────────────────────────────

def _run_quality_checks(
    mr: pd.DataFrame,
    sentiment: pd.DataFrame,
    pb: pd.DataFrame,
    merged: pd.DataFrame,
) -> None:
    checks = {
        "mr_rows": len(mr),
        "sentiment_rows": len(sentiment),
        "pb_rows": len(pb),
        "merged_rows": len(merged),
        "mr_missing_pct": round(merged["mr_raw_z"].isna().mean() * 100, 1),
        "news_missing_pct": round(merged["news_raw_velocity"].isna().mean() * 100, 1),
        "pb_missing_pct": round(merged["pb_rank"].isna().mean() * 100, 1),
    }
    logger.info("Data quality report", extra=checks)

    # Warn if coverage is very low
    for name, col in [("mr", "mr_raw_z"), ("news", "news_raw_velocity"), ("pb", "pb_rank")]:
        coverage = merged[col].notna().mean()
        if coverage < 0.5:
            logger.warning(
                f"Low factor coverage for {name}",
                extra={"factor": name, "coverage_pct": round(coverage * 100, 1)},
            )


# ── Pipeline ───────────────────────────────────────────────────────────────────

def generate_factor_scores(date: dt.date | None = None) -> pd.DataFrame:
    """
    Main entry point.

    Args:
        date: Trading date to compute factors for. Defaults to today.

    Returns:
        pd.DataFrame with columns: date, ticker, mr_z, news_z, pb_z
    """
    if date is None:
        from src.utils.date_utils import today_ny
        date = today_ny()

    cfg = get_settings()

    logger.info("=== Phase 2 Factor Pipeline START ===", extra={"date": str(date)})

    with LatencyTimer(logger, "full_factor_pipeline", date=str(date)):
        universe = _load_universe()
        logger.info("Universe loaded", extra={"tickers": len(universe)})

        # ── Factor 1: Mean Reversion ──────────────────────────────────────────
        logger.info("Computing Factor 1: Mean Reversion")
        mr_df = compute_mean_reversion_signal(date)

        # ── Factor 2: Sentiment Velocity ─────────────────────────────────────
        logger.info("Computing Factor 2: Sentiment Velocity")
        sentiment_df = compute_sentiment_velocity_signal(universe, date)

        # ── Factor 3: Price-to-Book ───────────────────────────────────────────
        logger.info("Computing Factor 3: Price-to-Book")
        pb_df = compute_pb_signal(universe, date)

        # ── Merge on ticker ───────────────────────────────────────────────────
        base = pd.DataFrame({"ticker": universe})
        merged = (
            base
            .merge(mr_df,       on="ticker", how="left")
            .merge(sentiment_df, on="ticker", how="left")
            .merge(pb_df,        on="ticker", how="left")
        )

        _run_quality_checks(mr_df, sentiment_df, pb_df, merged)

        # ── Winsorize raw signals ─────────────────────────────────────────────
        logger.info("Winsorizing raw signals")
        merged = winsorize_dataframe(
            merged,
            columns=["mr_raw_z", "news_raw_velocity", "pb_rank"],
            limits=cfg.normalization.winsorize_limits,
        )

        # ── Invert P/B rank: low rank (cheap) → high signal ──────────────────
        # We negate pb_rank so that a low percentile (cheap stock) gives a
        # high positive z-score, consistent with factor conventions.
        merged["pb_rank_inv"] = -merged["pb_rank"]

        # ── Cross-sectional Z-score normalization ─────────────────────────────
        logger.info("Applying cross-sectional Z-score normalization")
        merged = zscore_dataframe(
            merged,
            columns=["mr_raw_z", "news_raw_velocity", "pb_rank_inv"],
        )

        # ── Rename to final schema ────────────────────────────────────────────
        merged = merged.rename(columns={
            "mr_raw_z":          "mr_z",
            "news_raw_velocity": "news_z",
            "pb_rank_inv":       "pb_z",
        })

        # ── Add date column ───────────────────────────────────────────────────
        merged["date"] = pd.Timestamp(date)

        # ── Select and order final columns ────────────────────────────────────
        output = merged[["date", "ticker", "mr_z", "news_z", "pb_z"]].copy()

        # ── Validate schema ───────────────────────────────────────────────────
        try:
            validate_factor_output(output)
            logger.info("Schema validation passed")
        except Exception as exc:
            logger.error("Schema validation failed", extra={"error": str(exc)})
            raise

        # ── Persist to parquet ────────────────────────────────────────────────
        _save_factors(output, date)

        logger.info(
            "=== Phase 2 Factor Pipeline COMPLETE ===",
            extra={
                "date": str(date),
                "tickers_output": len(output),
                "mr_coverage": output["mr_z"].notna().sum(),
                "news_coverage": output["news_z"].notna().sum(),
                "pb_coverage": output["pb_z"].notna().sum(),
            },
        )

    return output


# ── Storage ────────────────────────────────────────────────────────────────────

def _save_factors(df: pd.DataFrame, date: dt.date) -> None:
    """
    Append today's factor scores to the partitioned parquet store.
    Partitioned by date for fast time-range queries.
    """
    cfg = get_settings()
    base_path = Path(cfg.storage.factors_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    # Partition path: data/features/factors/date=2026-03-10/factors.parquet
    partition_dir = base_path.parent / f"date={date.isoformat()}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    out_path = partition_dir / "factors.parquet"

    df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
    logger.info("Factors saved", extra={"path": str(out_path), "rows": len(df)})


def load_factor_scores(
    start: dt.date,
    end: dt.date,
) -> pd.DataFrame:
    """
    Load factor scores for a date range from the parquet store.
    Returns empty DataFrame if no data found.
    """
    cfg = get_settings()
    base_dir = Path(cfg.storage.factors_path).parent
    frames = []

    current = start
    while current <= end:
        partition = base_dir / f"date={current.isoformat()}" / "factors.parquet"
        if partition.exists():
            frames.append(pd.read_parquet(partition))
        current += dt.timedelta(days=1)

    if not frames:
        logger.warning("No factor data found", extra={"start": str(start), "end": str(end)})
        return pd.DataFrame(columns=["date", "ticker", "mr_z", "news_z", "pb_z"])

    return pd.concat(frames, ignore_index=True)
