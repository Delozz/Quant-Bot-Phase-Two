"""
P/B Ratio fetcher.
Thin wrapper that fetches fundamentals and validates P/B quality.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from src.ingestion.fundamentals_collector import get_fundamentals
from src.utils.logger import get_logger

logger = get_logger(__name__)


def fetch_pb_data(tickers: list[str], date: dt.date) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: ticker, pb, sector
    Only includes tickers with valid (positive) P/B ratios.
    """
    df = get_fundamentals(tickers, date)

    # Drop rows with invalid book value
    before = len(df)
    df = df[df["pb"] > 0].copy()
    dropped = before - len(df)
    if dropped:
        logger.warning(
            "Dropped tickers with non-positive P/B",
            extra={"dropped": dropped, "reason": "negative_or_zero_book_value"},
        )

    return df[["ticker", "pb", "sector"]].reset_index(drop=True)
