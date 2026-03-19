"""
Sector-neutral P/B ranking.

Banks typically trade at lower P/B than tech, so raw P/B
comparisons across sectors are misleading.

We rank each ticker within its own sector (percentile rank),
then invert so low P/B = high signal.
"""
from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Minimum tickers per sector to compute a meaningful rank.
# Tickers in sectors with fewer members are excluded.
MIN_SECTOR_SIZE = 3


def compute_sector_neutral_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:  DataFrame with columns  ticker, pb, sector
    Output: same DataFrame + column pb_rank  (0.0 → 1.0)

    pb_rank = percentile of pb within sector (ascending).
    Low  pb_rank → cheap within sector.
    """
    sector_counts = df.groupby("sector")["ticker"].count()
    valid_sectors = sector_counts[sector_counts >= MIN_SECTOR_SIZE].index
    excluded = df[~df["sector"].isin(valid_sectors)]

    if not excluded.empty:
        logger.warning(
            "Tickers excluded due to small sector size",
            extra={
                "excluded_tickers": excluded["ticker"].tolist(),
                "min_sector_size": MIN_SECTOR_SIZE,
            },
        )

    df = df[df["sector"].isin(valid_sectors)].copy()

    df["pb_rank"] = df.groupby("sector")["pb"].rank(pct=True, ascending=True)

    # Log sector distribution
    for sector, grp in df.groupby("sector"):
        logger.debug(
            "Sector rank computed",
            extra={
                "sector": sector,
                "count": len(grp),
                "median_pb": round(grp["pb"].median(), 2),
            },
        )

    return df
