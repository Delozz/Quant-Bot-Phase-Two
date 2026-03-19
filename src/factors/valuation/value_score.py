"""
Price-to-Book Value Factor Signal (Factor 3).

Signal: pb_rank percentile within sector (ascending).
Low P/B relative to sector peers = strong value signal.

Trigger: pb_rank <= 0.20  (bottom 20% in sector)
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from src.factors.valuation.pb_ratio_fetcher import fetch_pb_data
from src.factors.valuation.sector_ranker import compute_sector_neutral_rank
from src.utils.config import get_settings
from src.utils.logger import get_logger, LatencyTimer

logger = get_logger(__name__)


def compute_pb_signal(tickers: list[str], date: dt.date) -> pd.DataFrame:
    """
    Returns DataFrame with columns: ticker, pb_rank
    pb_rank in [0, 1], sector-neutral percentile.

    The normalization layer converts pb_rank → pb_z (inverted z-score)
    so that lower P/B → more positive signal.
    """
    cfg = get_settings().factors.pb_ratio

    with LatencyTimer(logger, "pb_signal", date=str(date), tickers=len(tickers)):
        # Fetch P/B + sector
        df = fetch_pb_data(tickers, date)

        # Sector-neutral rank
        df = compute_sector_neutral_rank(df)

        triggered = (df["pb_rank"] <= cfg.bottom_pct_threshold).sum()
        logger.info(
            "P/B signal computed",
            extra={
                "date": str(date),
                "tickers_with_signal": len(df),
                "triggered_bottom_20pct": int(triggered),
            },
        )

        return df[["ticker", "pb_rank"]].reset_index(drop=True)
