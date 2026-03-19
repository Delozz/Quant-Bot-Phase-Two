"""
Mean Reversion Factor Signal (Factor 1).

Formula:
    Z = (residual_today - rolling_mean) / rolling_std
    Z clipped to [-5, 5]

Trigger condition: Z <= -2.0  (2σ below normal)
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from src.factors.mean_reversion.residual_loader import load_residuals
from src.factors.mean_reversion.rolling_stats import compute_rolling_stats
from src.utils.config import get_settings
from src.utils.logger import get_logger, LatencyTimer

logger = get_logger(__name__)


def compute_mean_reversion_signal(date: dt.date) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: ticker, mr_raw_z
    filtered to the single `date`, with z-scores clipped.

    The caller (normalization layer) applies final cross-sectional z-scoring.
    """
    cfg = get_settings().factors.mean_reversion

    with LatencyTimer(logger, "mean_reversion_signal", date=str(date)):
        # Load rolling window of residuals
        residuals = load_residuals(date, lookback_days=cfg.rolling_window * 2)

        # Compute rolling stats across full history
        df = compute_rolling_stats(residuals)

        # Keep only today's row per ticker
        df["date"] = pd.to_datetime(df["date"]).dt.date
        today_df = df[df["date"] == date].copy()

        if today_df.empty:
            logger.warning("No residuals for date", extra={"date": str(date)})
            return pd.DataFrame(columns=["ticker", "mr_raw_z"])

        # Compute z-score
        today_df["mr_raw_z"] = (
            (today_df["residual"] - today_df["rolling_mean"]) / today_df["rolling_std"]
        )

        # Clip extremes
        lo, hi = cfg.clip_range
        today_df["mr_raw_z"] = today_df["mr_raw_z"].clip(lo, hi)

        # Log trigger count
        triggered = (today_df["mr_raw_z"] <= cfg.z_trigger).sum()
        logger.info(
            "Mean reversion signal computed",
            extra={
                "date": str(date),
                "tickers": len(today_df),
                "triggered": int(triggered),
                "trigger_threshold": cfg.z_trigger,
            },
        )

        return today_df[["ticker", "mr_raw_z"]].dropna().reset_index(drop=True)
