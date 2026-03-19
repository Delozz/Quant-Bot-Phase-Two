"""
Cross-sectional Z-score normalization.

z = (x - mean) / std

Applied across the universe on each date so that factors are
comparable across tickers.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def zscore_series(series: pd.Series, min_std: float = 1e-8) -> pd.Series:
    """
    Compute cross-sectional z-score for a single Series.
    NaN values are excluded from mean/std calculation and remain NaN.

    Args:
        series:  Numeric pandas Series.
        min_std: Minimum std threshold — prevents division by near-zero std.
    """
    mu = series.mean(skipna=True)
    sigma = series.std(skipna=True)

    if pd.isna(sigma) or sigma < min_std:
        logger.warning(
            "Near-zero std in z-score normalization — returning zeros",
            extra={"column": series.name, "std": float(sigma) if not pd.isna(sigma) else None},
        )
        return pd.Series(np.where(series.notna(), 0.0, np.nan), index=series.index, name=series.name)

    return (series - mu) / sigma


def zscore_dataframe(
    df: pd.DataFrame,
    columns: list[str],
    min_std: float = 1e-8,
) -> pd.DataFrame:
    """Apply cross-sectional z-scoring to multiple columns. Returns a copy."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = zscore_series(df[col], min_std=min_std)
    return df
