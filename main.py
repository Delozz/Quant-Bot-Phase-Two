"""
QuantBot Phase 2 — CLI Entry Point

Usage:
    # Run pipeline for today
    python main.py

    # Run for a specific date
    python main.py --date 2026-03-10

    # Run in scheduled daemon mode (runs every trading day at 18:30 EST)
    python main.py --schedule

    # Load and display saved factors
    python main.py --load --start 2026-03-01 --end 2026-03-14

    # Run test suite
    python main.py --test
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time

import pandas as pd

from src.pipeline.factor_pipeline import generate_factor_scores, load_factor_scores
from src.utils.config import get_settings
from src.utils.date_utils import today_ny, is_trading_day, NY_TZ
from src.utils.logger import get_logger

logger = get_logger(
    __name__,
    level=get_settings().logging.level,
    log_file=get_settings().logging.output,
)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QuantBot Phase 2 — Factor Engine")
    parser.add_argument("--date",     type=str,  help="Date to run (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--schedule", action="store_true", help="Run as a daily scheduler daemon")
    parser.add_argument("--load",     action="store_true", help="Load and display saved factor scores")
    parser.add_argument("--start",    type=str,  help="Start date for --load (YYYY-MM-DD)")
    parser.add_argument("--end",      type=str,  help="End date for --load (YYYY-MM-DD)")
    parser.add_argument("--test",     action="store_true", help="Run test suite")
    return parser.parse_args()


# ── Pretty printer ─────────────────────────────────────────────────────────────

def print_factor_table(df: pd.DataFrame) -> None:
    if df.empty:
        print("No factor data to display.")
        return

    pd.set_option("display.max_rows", 60)
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 120)

    print("\n" + "=" * 80)
    print(f"  Phase 2 Factor Scores  |  {df['date'].iloc[0].date()}  |  {len(df)} tickers")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

    # Summary stats
    print("\nFactor Summary Statistics:")
    print(df[["mr_z", "news_z", "pb_z"]].describe().round(4).to_string())

    # Highlight mean reversion candidates
    mr_candidates = df[df["mr_z"] <= -2.0][["ticker", "mr_z", "news_z", "pb_z"]]
    if not mr_candidates.empty:
        print(f"\n🎯 Mean Reversion Candidates (mr_z ≤ -2.0): {len(mr_candidates)} tickers")
        print(mr_candidates.to_string(index=False))
    print()


# ── Scheduler ─────────────────────────────────────────────────────────────────

def run_scheduler() -> None:
    """
    Daemon mode: waits until 18:30 EST each trading day then runs the pipeline.
    Blocks indefinitely. Run with `nohup` or as a systemd service in production.
    """
    cfg = get_settings()
    run_hour, run_minute = map(int, cfg.scheduling.get("run_time", "18:30").split(":"))

    logger.info("Scheduler started", extra={"run_time": f"{run_hour:02d}:{run_minute:02d} EST"})
    print(f"⏰  Scheduler running — pipeline fires daily at {run_hour:02d}:{run_minute:02d} EST on trading days.")
    print("    Press Ctrl+C to stop.\n")

    last_run_date: dt.date | None = None

    try:
        while True:
            now_est = dt.datetime.now(tz=NY_TZ)
            today = now_est.date()

            if (
                is_trading_day(today)
                and today != last_run_date
                and now_est.hour >= run_hour
                and now_est.minute >= run_minute
            ):
                logger.info("Scheduler triggered pipeline", extra={"date": str(today)})
                try:
                    scores = generate_factor_scores(today)
                    print_factor_table(scores)
                    last_run_date = today
                except Exception as exc:
                    logger.error("Pipeline run failed", extra={"error": str(exc)})

            time.sleep(30)  # check every 30 seconds

    except KeyboardInterrupt:
        print("\nScheduler stopped.")
        logger.info("Scheduler stopped by user")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Run tests
    if args.test:
        import pytest
        sys.exit(pytest.main(["-v", "tests/"]))

    # Load saved factor scores
    if args.load:
        start = dt.date.fromisoformat(args.start) if args.start else today_ny() - dt.timedelta(days=7)
        end   = dt.date.fromisoformat(args.end)   if args.end   else today_ny()
        df = load_factor_scores(start, end)
        print_factor_table(df)
        return

    # Scheduled daemon
    if args.schedule:
        run_scheduler()
        return

    # Single run
    date = dt.date.fromisoformat(args.date) if args.date else today_ny()
    scores = generate_factor_scores(date)
    print_factor_table(scores)


if __name__ == "__main__":
    main()
