"""
Winsorization: clips extreme percentile values to reduce outlier influence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize as scipy_winsorize

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def winsorize_series(
    series: pd.Series,
    limits: list[float] | None = None,
) -> pd.Series:
    """
    Clips values below the `limits[0]` percentile and above the `1 - limits[1]` percentile.
    NaN values are preserved.

    Args:
        series: The numeric series to winsorize.
        limits: [lower_pct, upper_pct], e.g. [0.01, 0.01] clips 1% each tail.
    """
    if limits is None:
        limits = get_settings().normalization.winsorize_limits

    mask = series.notna()
    if mask.sum() < 4:
        # Not enough data to winsorize meaningfully
        return series

    winsorized = scipy_winsorize(series[mask].values, limits=limits)
    result = series.copy()
    result[mask] = winsorized

    clipped_lo = int((series < result).sum())
    clipped_hi = int((series > result).sum())
    if clipped_lo + clipped_hi > 0:
        logger.debug(
            "Winsorization clipped values",
            extra={
                "column": series.name,
                "clipped_low": clipped_lo,
                "clipped_high": clipped_hi,
            },
        )
    return result


def winsorize_dataframe(
    df: pd.DataFrame,
    columns: list[str],
    limits: list[float] | None = None,
) -> pd.DataFrame:
    """Apply winsorization to multiple columns in place (returns copy)."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = winsorize_series(df[col], limits=limits)
    return df
