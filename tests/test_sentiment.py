"""
Tests for Sentiment Velocity Factor (Factor 2).
"""
import datetime as dt

import pytest

from src.ingestion.news_collector import NewsArticle, clean_text, deduplicate
from src.factors.sentiment_velocity.sentiment_aggregator import (
    aggregate_sentiment,
    _decay_weight,
)


# ── Text cleaning tests ───────────────────────────────────────────────────────

class TestCleanText:
    def test_html_stripped(self):
        assert "<b>" not in clean_text("<b>Earnings beat</b>")

    def test_boilerplate_removed(self):
        result = clean_text("This press release contains forward-looking statements.")
        assert "forward-looking" not in result.lower()

    def test_whitespace_collapsed(self):
        result = clean_text("Too   many    spaces")
        assert "  " not in result


class TestDeduplicate:
    def _make_article(self, headline: str) -> NewsArticle:
        return NewsArticle(
            ticker="AAPL",
            headline=headline,
            body="",
            timestamp=dt.datetime(2026, 3, 10, 12, 0, tzinfo=dt.timezone.utc),
            source="Reuters",
            relevance_score=0.9,
        )

    def test_duplicates_removed(self):
        articles = [self._make_article("Apple beats earnings")] * 3
        result = deduplicate(articles)
        assert len(result) == 1

    def test_unique_kept(self):
        articles = [
            self._make_article("Apple beats earnings"),
            self._make_article("Apple misses revenue"),
        ]
        result = deduplicate(articles)
        assert len(result) == 2


# ── Decay weight tests ────────────────────────────────────────────────────────

class TestDecayWeight:
    def test_recent_article_weight_near_one(self):
        now = dt.datetime(2026, 3, 10, 12, 0, tzinfo=dt.timezone.utc)
        article_time = now - dt.timedelta(minutes=30)
        w = _decay_weight(article_time, now, lam=0.1)
        assert w > 0.99

    def test_old_article_weight_small(self):
        now = dt.datetime(2026, 3, 10, 12, 0, tzinfo=dt.timezone.utc)
        article_time = now - dt.timedelta(hours=48)
        w = _decay_weight(article_time, now, lam=0.1)
        assert w < 0.01

    def test_weight_monotone_decreasing(self):
        now = dt.datetime(2026, 3, 10, 12, 0, tzinfo=dt.timezone.utc)
        weights = [
            _decay_weight(now - dt.timedelta(hours=h), now, lam=0.1)
            for h in [1, 6, 12, 24, 48]
        ]
        assert all(weights[i] > weights[i + 1] for i in range(len(weights) - 1))


# ── Sentiment aggregator tests ────────────────────────────────────────────────

class TestSentimentAggregation:
    def _make_articles(self, count: int = 10, hours_old: int = 2) -> list[NewsArticle]:
        now = dt.datetime(2026, 3, 10, 18, 0, tzinfo=dt.timezone.utc)
        articles = []
        for i in range(count):
            articles.append(
                NewsArticle(
                    ticker="TSLA",
                    headline=f"Company news headline number {i}",
                    body="",
                    timestamp=now - dt.timedelta(hours=hours_old + i),
                    source="Bloomberg",
                    relevance_score=0.9,
                )
            )
        return articles

    def test_empty_articles_returns_zeros(self):
        result = aggregate_sentiment([])
        assert result["velocity"] == 0.0
        assert result["article_count"] == 0

    def test_output_keys_present(self):
        articles = self._make_articles(5)
        result = aggregate_sentiment(articles)
        assert "sentiment_24h" in result
        assert "sentiment_7d" in result
        assert "velocity" in result
        assert "article_count" in result

    def test_velocity_is_difference(self):
        articles = self._make_articles(10)
        result = aggregate_sentiment(articles)
        assert abs(result["velocity"] - (result["sentiment_24h"] - result["sentiment_7d"])) < 1e-9
