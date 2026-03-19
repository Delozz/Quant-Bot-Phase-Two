"""
LSEG / Refinitiv Data API client.

Wraps lseg-data library with:
  - automatic session management
  - retry + rate-limit handling
  - yfinance fallback for price/fundamentals
"""
from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.utils.logger import get_logger, LatencyTimer
from src.utils.config import get_api_keys

logger = get_logger(__name__)

# ── Session singleton ──────────────────────────────────────────────────────────

_session_open = False


def open_session() -> None:
    global _session_open
    if _session_open:
        return
    keys = get_api_keys().lseg
    try:
        import lseg.data as ld
        ld.open_session(
            app_key=keys.app_key,
            username=keys.username,
            password=keys.password,
        )
        _session_open = True
        logger.info("LSEG session opened")
    except Exception as exc:
        logger.warning("LSEG session failed — will use yfinance fallback", extra={"error": str(exc)})


def close_session() -> None:
    global _session_open
    if not _session_open:
        return
    try:
        import lseg.data as ld
        ld.close_session()
        _session_open = False
        logger.info("LSEG session closed")
    except Exception:
        pass


# ── Price history ──────────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(Exception),
)
def fetch_price_history(
    tickers: list[str],
    start: dt.date,
    end: dt.date,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: date, ticker, close, volume.
    Falls back to yfinance if LSEG is unavailable.
    """
    with LatencyTimer(logger, "fetch_price_history", tickers_count=len(tickers)):
        if _session_open:
            return _lseg_price_history(tickers, start, end)
        return _yfinance_price_history(tickers, start, end)


def _lseg_price_history(tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    import lseg.data as ld
    raw = ld.get_history(
        universe=tickers,
        fields=["CLOSE", "VOLUME"],
        start=str(start),
        end=str(end),
    )
    df = raw.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"date": "date", "close": "close", "volume": "volume"})
    df["ticker"] = df.get("instrument", df.get("ticker", ""))
    return df[["date", "ticker", "close", "volume"]].dropna()


def _yfinance_price_history(tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(
        tickers,
        start=str(start),
        end=str(end),
        auto_adjust=True,
        progress=False,
    )
    rows = []
    close = raw["Close"] if len(tickers) > 1 else raw[["Close"]].rename(columns={"Close": tickers[0]})
    volume = raw["Volume"] if len(tickers) > 1 else raw[["Volume"]].rename(columns={"Volume": tickers[0]})
    for ticker in tickers:
        if ticker not in close.columns:
            continue
        tmp = pd.DataFrame({
            "date": close.index.date,
            "ticker": ticker,
            "close": close[ticker].values,
            "volume": volume[ticker].values,
        })
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True).dropna() if rows else pd.DataFrame()


# ── Fundamentals ───────────────────────────────────────────────────────────────

def fetch_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """
    Returns DataFrame with: ticker, price, book_value_per_share, sector.
    Falls back to yfinance if LSEG unavailable.
    """
    with LatencyTimer(logger, "fetch_fundamentals", tickers_count=len(tickers)):
        if _session_open:
            return _lseg_fundamentals(tickers)
        return _yfinance_fundamentals(tickers)


def _lseg_fundamentals(tickers: list[str]) -> pd.DataFrame:
    import lseg.data as ld
    raw = ld.get_data(
        universe=tickers,
        fields=["TR.PriceToBook", "TR.BookValuePerShare", "TR.TRBCEconomicSector"],
    )
    df = raw.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={
        "tr.pricetobook": "pb",
        "tr.bookvaluepershare": "book_value_per_share",
        "tr.trbceconomicsector": "sector",
        "instrument": "ticker",
    })
    return df[["ticker", "pb", "book_value_per_share", "sector"]].dropna(subset=["ticker"])


def _yfinance_fundamentals(tickers: list[str]) -> pd.DataFrame:
    import yfinance as yf
    rows = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            bvps = info.get("bookValue")
            sector = info.get("sector", "Unknown")
            pb = (price / bvps) if price and bvps and bvps != 0 else None
            rows.append({
                "ticker": ticker,
                "pb": pb,
                "book_value_per_share": bvps,
                "sector": sector,
            })
        except Exception as exc:
            logger.warning("yfinance fundamentals failed", extra={"ticker": ticker, "error": str(exc)})
    return pd.DataFrame(rows)
