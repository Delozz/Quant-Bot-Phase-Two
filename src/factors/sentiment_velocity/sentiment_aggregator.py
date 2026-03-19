"""
Sentiment aggregator.

Takes a list of NewsArticle objects and computes:
  - Sentiment_24h  (short window, decay-weighted)
  - Sentiment_7d   (long window,  decay-weighted)
  - Velocity       = Sentiment_24h - Sentiment_7d
"""
from __future__ import annotations

import datetime as dt
import math
from typing import Sequence

import pandas as pd

from src.ingestion.news_collector import NewsArticle
from src.factors.sentiment_velocity.finnbert_classifier import classify_batch
from src.utils.config import get_settings
from src.utils.date_utils import now_utc
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _decay_weight(article_time: dt.datetime, now: dt.datetime, lam: float) -> float:
    """
    Exponential decay:  w = e^(-λ * age_in_hours)
    Recent articles receive weight close to 1; older ones decay toward 0.
    """
    age_hours = max(0.0, (now - article_time).total_seconds() / 3600)
    return math.exp(-lam * age_hours)


def _weighted_mean(numerics: list[float], weights: list[float]) -> float:
    if not numerics or sum(weights) == 0:
        return 0.0
    total_w = sum(weights)
    return sum(n * w for n, w in zip(numerics, weights)) / total_w


def aggregate_sentiment(
    articles: list[NewsArticle],
    now: dt.datetime | None = None,
) -> dict[str, float]:
    """
    Returns:
        {
          "sentiment_24h":  float,
          "sentiment_7d":   float,
          "velocity":       float,
          "article_count":  int,
        }
    """
    cfg = get_settings().factors.sentiment
    now = now or now_utc()
    lam = cfg.decay_lambda

    if not articles:
        return {"sentiment_24h": 0.0, "sentiment_7d": 0.0, "velocity": 0.0, "article_count": 0}

    # Run all headlines through FinnBERT in one pass
    headlines = [a.headline for a in articles]
    sentiments = classify_batch(headlines)

    now_24h_cutoff = now - dt.timedelta(hours=cfg.short_window_hours)
    now_7d_cutoff  = now - dt.timedelta(days=cfg.long_window_days)

    art_24h_nums, art_24h_wts = [], []
    art_7d_nums,  art_7d_wts  = [], []

    for article, sent in zip(articles, sentiments):
        ts = article.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=dt.timezone.utc)
        relevance_weight = article.relevance_score  # source quality multiplier

        if ts >= now_24h_cutoff:
            w = _decay_weight(ts, now, lam) * relevance_weight
            art_24h_nums.append(sent.numeric)
            art_24h_wts.append(w)

        if ts >= now_7d_cutoff:
            w = _decay_weight(ts, now, lam) * relevance_weight
            art_7d_nums.append(sent.numeric)
            art_7d_wts.append(w)

    s24 = _weighted_mean(art_24h_nums, art_24h_wts)
    s7d = _weighted_mean(art_7d_nums,  art_7d_wts)
    velocity = s24 - s7d

    logger.debug(
        "Sentiment aggregated",
        extra={
            "articles_24h": len(art_24h_nums),
            "articles_7d": len(art_7d_nums),
            "sentiment_24h": round(s24, 4),
            "sentiment_7d": round(s7d, 4),
            "velocity": round(velocity, 4),
        },
    )
    return {
        "sentiment_24h": s24,
        "sentiment_7d": s7d,
        "velocity": velocity,
        "article_count": len(articles),
    }


def aggregate_universe_sentiment(
    news_by_ticker: dict[str, list[NewsArticle]],
    min_articles: int | None = None,
) -> pd.DataFrame:
    """
    Runs sentiment aggregation for every ticker in the news dict.

    Returns DataFrame with columns:
        ticker, sentiment_24h, sentiment_7d, velocity, article_count
    """
    cfg = get_settings().factors.sentiment
    min_articles = min_articles or cfg.min_articles
    now = now_utc()
    rows = []

    for ticker, articles in news_by_ticker.items():
        result = aggregate_sentiment(articles, now=now)
        result["ticker"] = ticker
        result["has_min_coverage"] = result["article_count"] >= min_articles
        rows.append(result)

    df = pd.DataFrame(rows)
    covered = df["has_min_coverage"].sum()
    logger.info(
        "Universe sentiment aggregated",
        extra={"tickers": len(df), "with_coverage": int(covered)},
    )
    return df
