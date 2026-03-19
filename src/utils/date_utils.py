"""
Date/trading-calendar helpers for QuantBot.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd

NY_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")


def today_ny() -> dt.date:
    """Return today's date in New York time."""
    return dt.datetime.now(tz=NY_TZ).date()


def now_utc() -> dt.datetime:
    return dt.datetime.now(tz=UTC_TZ)


def is_trading_day(date: dt.date) -> bool:
    """
    Simple weekday check. For production, swap in a proper exchange-calendar
    library (e.g. `exchange_calendars`) to handle US market holidays.
    """
    return date.weekday() < 5  # Mon–Fri


def previous_trading_day(date: dt.date) -> dt.date:
    candidate = date - dt.timedelta(days=1)
    while not is_trading_day(candidate):
        candidate -= dt.timedelta(days=1)
    return candidate


def trading_days_between(start: dt.date, end: dt.date) -> list[dt.date]:
    """Return all trading days in [start, end] inclusive."""
    days = pd.bdate_range(start=start, end=end)
    return [d.date() for d in days]


def window_start_utc(hours: int | None = None, days: int | None = None) -> dt.datetime:
    """Return a UTC datetime `hours` or `days` ago from now."""
    now = now_utc()
    if hours is not None:
        return now - dt.timedelta(hours=hours)
    if days is not None:
        return now - dt.timedelta(days=days)
    raise ValueError("Provide either hours or days.")


def to_date(value: str | dt.date | dt.datetime) -> dt.date:
    """Coerce various date-like inputs to a plain `date`."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value))
