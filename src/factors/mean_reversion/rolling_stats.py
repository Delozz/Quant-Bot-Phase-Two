"""
Computes rolling mean and standard deviation of PCA residuals per ticker.
"""
from __future__ import annotations

import pandas as pd

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects columns: date, ticker, residual
    Adds columns:    rolling_mean, rolling_std

    Uses a per-ticker rolling window to preserve independence across tickers.
    """
    cfg = get_settings().factors.mean_reversion
    window = cfg.rolling_window

    df = df.sort_values(["ticker", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"])

    df["rolling_mean"] = (
        df.groupby("ticker")["residual"]
        .transform(lambda s: s.rolling(window, min_periods=max(1, window // 2)).mean())
    )
    df["rolling_std"] = (
        df.groupby("ticker")["residual"]
        .transform(lambda s: s.rolling(window, min_periods=max(1, window // 2)).std())
    )

    # Guard against zero/NaN std (flat residual series)
    zero_std = (df["rolling_std"] == 0) | df["rolling_std"].isna()
    if zero_std.any():
        logger.warning(
            "Zero or NaN rolling_std detected — will produce NaN z-scores for affected rows",
            extra={"rows_affected": int(zero_std.sum())},
        )

    return df
