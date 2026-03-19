"""
Async news collector.

Fetches headlines + bodies from LSEG News API for a list of tickers
using asyncio.gather() for parallel requests.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.logger import get_logger, LatencyTimer
from src.utils.cache import Cache
from src.utils.config import get_api_keys, get_settings
from src.utils.date_utils import window_start_utc

logger = get_logger(__name__)
_cache = Cache(prefix="news")

# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class NewsArticle:
    ticker: str
    headline: str
    body: str
    timestamp: dt.datetime
    source: str
    relevance_score: float


# ── Text cleaning ──────────────────────────────────────────────────────────────

_BOILERPLATE = re.compile(
    r"(safe harbor|forward.looking statements|this press release|"
    r"for immediate release|media contact|investor relations)",
    re.IGNORECASE,
)
_HTML_TAG = re.compile(r"<[^>]+>")
_TICKER_PATTERN = re.compile(r"\b[A-Z]{1,5}\b")  # crude ticker remover in headlines


def clean_text(text: str) -> str:
    """Remove HTML, tickers, duplicate whitespace, boilerplate."""
    text = _HTML_TAG.sub(" ", text)
    text = _BOILERPLATE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def deduplicate(articles: list[NewsArticle]) -> list[NewsArticle]:
    seen: set[str] = set()
    out: list[NewsArticle] = []
    for a in articles:
        key = a.headline.lower().strip()
        if key not in seen:
            seen.add(key)
            out.append(a)
    return out


# ── LSEG News fetcher ──────────────────────────────────────────────────────────

class LSEGNewsClient:
    """
    Async wrapper around LSEG News REST endpoint.
    Falls back to an empty list if credentials are missing.
    """

    BASE_URL = "https://api.refinitiv.com/message-store/beta1/news/headlines"

    def __init__(self):
        keys = get_api_keys().lseg
        self.app_key = keys.app_key
        self.headers = {
            "X-ApplicationId": self.app_key,
            "Content-Type": "application/json",
        }
        cfg = get_settings().factors.sentiment
        self.min_relevance = cfg.min_relevance

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _fetch_ticker(
        self,
        session: aiohttp.ClientSession,
        ticker: str,
        since: dt.datetime,
    ) -> list[dict[str, Any]]:
        params = {
            "query": ticker,
            "dateFrom": since.isoformat(),
            "limit": 100,
        }
        async with session.get(self.BASE_URL, headers=self.headers, params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("data", {}).get("headlines", [])
            logger.warning(
                "LSEG news request failed",
                extra={"ticker": ticker, "status": resp.status},
            )
            return []

    async def fetch_all(
        self,
        tickers: list[str],
        hours_back: int = 168,  # 7 days
    ) -> dict[str, list[NewsArticle]]:
        since = window_start_utc(hours=hours_back)
        results: dict[str, list[NewsArticle]] = {t: [] for t in tickers}

        if not self.app_key or self.app_key.startswith("YOUR_"):
            logger.warning("LSEG app_key not configured — returning empty news")
            return results

        connector = aiohttp.TCPConnector(limit=20)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = {ticker: self._fetch_ticker(session, ticker, since) for ticker in tickers}
            raw_results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        for ticker, raw in zip(tickers, raw_results):
            if isinstance(raw, Exception):
                logger.error("News fetch error", extra={"ticker": ticker, "error": str(raw)})
                continue
            articles = []
            for item in raw:
                relevance = float(item.get("relevance", 0.0))
                if relevance < self.min_relevance:
                    continue
                articles.append(
                    NewsArticle(
                        ticker=ticker,
                        headline=clean_text(item.get("headline", {}).get("text", "")),
                        body=clean_text(item.get("storyId", "")),  # body fetched separately in prod
                        timestamp=dt.datetime.fromisoformat(
                            item.get("firstCreated", since.isoformat())
                        ),
                        source=item.get("sourceCode", ""),
                        relevance_score=relevance,
                    )
                )
            results[ticker] = deduplicate(articles)
            logger.info(
                "News fetched",
                extra={"ticker": ticker, "article_count": len(results[ticker])},
            )
        return results


# ── Public API ─────────────────────────────────────────────────────────────────

async def collect_news(
    tickers: list[str],
    hours_back: int = 168,
    use_cache: bool = True,
) -> dict[str, list[NewsArticle]]:
    """
    Fetch and cache news for a list of tickers.
    Returns dict mapping ticker → list[NewsArticle].
    """
    cache_key = f"news:{','.join(sorted(tickers))}:h{hours_back}"
    if use_cache:
        cached = _cache.get(cache_key)
        if cached is not None:
            logger.info("News cache hit", extra={"key": cache_key})
            return cached

    with LatencyTimer(logger, "collect_news", tickers_count=len(tickers)):
        client = LSEGNewsClient()
        data = await client.fetch_all(tickers, hours_back=hours_back)

    if use_cache:
        _cache.set(cache_key, data, ttl=1800)  # 30 min TTL
    return data
