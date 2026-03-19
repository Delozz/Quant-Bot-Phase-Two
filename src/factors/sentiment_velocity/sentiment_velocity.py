"""
Sentiment Velocity Factor Signal (Factor 2).

velocity = Sentiment_24h − Sentiment_7d

Cross-sectionally z-scored across the universe on each date.
Tickers with fewer than min_articles are excluded (NaN).
"""
from __future__ import annotations

import asyncio
import datetime as dt

import pandas as pd

from src.ingestion.news_collector import collect_news
from src.factors.sentiment_velocity.sentiment_aggregator import aggregate_universe_sentiment
from src.utils.config import get_settings
from src.utils.logger import get_logger, LatencyTimer

logger = get_logger(__name__)


def compute_sentiment_velocity_signal(
    tickers: list[str],
    date: dt.date,
) -> pd.DataFrame:
    """
    Returns DataFrame with columns: ticker, news_raw_velocity
    Rows with insufficient news coverage are dropped.
    """
    cfg = get_settings().factors.sentiment

    with LatencyTimer(logger, "sentiment_velocity_signal", date=str(date), tickers=len(tickers)):
        # Collect news (7-day window for full velocity calculation)
        news_by_ticker = asyncio.run(
            collect_news(tickers, hours_back=cfg.long_window_days * 24)
        )

        # Aggregate sentiment + compute velocity
        df = aggregate_universe_sentiment(news_by_ticker, min_articles=cfg.min_articles)

        # Drop tickers without sufficient coverage
        original_len = len(df)
        df = df[df["has_min_coverage"]].copy()
        dropped = original_len - len(df)
        if dropped:
            logger.warning(
                "Tickers dropped due to insufficient news coverage",
                extra={"dropped": dropped, "min_required": cfg.min_articles},
            )

        df = df.rename(columns={"velocity": "news_raw_velocity"})

        logger.info(
            "Sentiment velocity computed",
            extra={
                "date": str(date),
                "tickers_with_signal": len(df),
            },
        )
        return df[["ticker", "news_raw_velocity"]].reset_index(drop=True)
