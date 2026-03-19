"""
Loads PCA residuals produced by Phase 1.

Expected parquet schema:
    date     : date
    ticker   : str
    residual : float64
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_residuals(
    date: dt.date,
    lookback_days: int = 60,
) -> pd.DataFrame:
    """
    Returns a DataFrame of PCA residuals for the rolling window ending on `date`.

    Columns: date, ticker, residual
    """
    path = Path(get_settings().storage.residuals_path)

    if not path.exists():
        logger.warning(
            "PCA residuals file not found — generating synthetic residuals for development",
            extra={"path": str(path)},
        )
        return _synthetic_residuals(date, lookback_days)

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    end_date = date
    start_date = date - dt.timedelta(days=lookback_days)
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

    if df.empty:
        logger.warning("No residuals found in date range", extra={"start": str(start_date), "end": str(end_date)})

    logger.info(
        "Residuals loaded",
        extra={"rows": len(df), "tickers": df["ticker"].nunique(), "date_range": f"{start_date}:{end_date}"},
    )
    return df.reset_index(drop=True)


def _synthetic_residuals(date: dt.date, lookback_days: int) -> pd.DataFrame:
    """
    Generates synthetic residuals for development/testing.
    Replace this with real Phase 1 output in production.
    """
    import numpy as np

    SP500_SAMPLE = [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
        "BRK-B", "UNH", "LLY", "JPM", "XOM", "V", "AVGO", "PG",
        "MA", "HD", "CVX", "MRK", "ABBV", "AMD", "COST", "ADBE",
        "CRM", "ACN",
    ]
    rng = np.random.default_rng(42)
    rows = []
    for i in range(lookback_days):
        d = date - dt.timedelta(days=lookback_days - i)
        for ticker in SP500_SAMPLE:
            rows.append({
                "date": d,
                "ticker": ticker,
                "residual": rng.normal(0, 0.02),
            })
    return pd.DataFrame(rows)
