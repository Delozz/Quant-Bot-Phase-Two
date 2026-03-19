"""
Fundamentals collector.
Pulls P/B ratios, book value, and sector classification for the universe.
Caches results for the trading day.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from src.ingestion.lseg_client import fetch_fundamentals
from src.utils.cache import Cache
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
_cache = Cache(prefix="fundamentals")


def get_fundamentals(
    tickers: list[str],
    date: dt.date,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        ticker, pb, book_value_per_share, sector

    Applies basic quality filters:
      - drops rows where pb <= 0 (negative book value)
      - fills missing sector with 'Unknown'
    """
    cache_key = f"{date.isoformat()}:{','.join(sorted(tickers))}"
    if use_cache:
        cached = _cache.get(cache_key)
        if cached is not None:
            logger.info("Fundamentals cache hit", extra={"date": str(date)})
            return cached

    logger.info("Fetching fundamentals", extra={"tickers_count": len(tickers), "date": str(date)})
    df = fetch_fundamentals(tickers)

    # Quality filters
    df = df[df["pb"].notna()]
    df = df[df["pb"] > 0]
    df["sector"] = df["sector"].fillna("Unknown")
    df = df[df["ticker"].isin(tickers)]

    logger.info(
        "Fundamentals fetched",
        extra={"total": len(tickers), "valid": len(df), "dropped": len(tickers) - len(df)},
    )

    if use_cache:
        _cache.set(cache_key, df, ttl=86400)  # cache for 24h (fundamentals change slowly)
    return df.reset_index(drop=True)
