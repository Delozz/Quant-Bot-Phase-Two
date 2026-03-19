"""
Tests for Price-to-Book Value Factor (Factor 3).
"""
import pandas as pd
import numpy as np
import pytest

from src.factors.valuation.sector_ranker import compute_sector_neutral_rank


# ── Sector ranker tests ───────────────────────────────────────────────────────

class TestSectorNeutralRank:
    def _make_pb_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "ticker": ["JPM", "BAC", "WFC", "GS",      # Banks — low PB
                        "NVDA", "AMD", "INTC", "QCOM",  # Tech — high PB
                        "XOM", "CVX", "COP"],            # Energy — mid
            "pb": [1.1, 0.9, 1.0, 1.2,
                   25.0, 18.0, 12.0, 15.0,
                   2.1, 2.4, 1.8],
            "sector": ["Financials", "Financials", "Financials", "Financials",
                        "Technology", "Technology", "Technology", "Technology",
                        "Energy", "Energy", "Energy"],
        })

    def test_ranks_in_zero_one(self):
        df = self._make_pb_df()
        out = compute_sector_neutral_rank(df)
        assert out["pb_rank"].between(0, 1).all()

    def test_highest_pb_in_sector_has_rank_one(self):
        df = self._make_pb_df()
        out = compute_sector_neutral_rank(df)
        # Within Financials, GS has highest PB (1.2) → should have rank = 1.0
        gs_rank = out.loc[out["ticker"] == "GS", "pb_rank"].values[0]
        assert gs_rank == pytest.approx(1.0)

    def test_lowest_pb_in_sector_has_rank_near_zero(self):
        df = self._make_pb_df()
        out = compute_sector_neutral_rank(df)
        # Within Financials, BAC has lowest PB (0.9) → rank = 0.25 (1/4)
        bac_rank = out.loc[out["ticker"] == "BAC", "pb_rank"].values[0]
        assert bac_rank == pytest.approx(0.25)

    def test_sector_independent_ranking(self):
        """A bank with PB=1.2 should rank differently than a tech with PB=1.2."""
        df = self._make_pb_df()
        out = compute_sector_neutral_rank(df)
        # GS (Financials, PB=1.2) is rank 1.0 in Financials
        # INTC (Technology, PB=12.0) is rank ~0.25 in Tech
        # They're not cross-contaminating each other
        gs_rank = out.loc[out["ticker"] == "GS", "pb_rank"].values[0]
        intc_rank = out.loc[out["ticker"] == "INTC", "pb_rank"].values[0]
        assert gs_rank != intc_rank

    def test_small_sectors_excluded(self):
        """Sectors with fewer than MIN_SECTOR_SIZE tickers should be dropped."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT", "GOOG",  # big sector (3)
                        "LONE"],                  # singleton sector (1) — excluded
            "pb": [30.0, 28.0, 25.0, 5.0],
            "sector": ["Technology", "Technology", "Technology", "Singleton"],
        })
        out = compute_sector_neutral_rank(df)
        assert "LONE" not in out["ticker"].values
        assert len(out) == 3

    def test_negative_pb_not_in_input(self):
        """Negative PB should have been filtered before reaching this function.
        Ranker should handle it gracefully if it slips through."""
        df = pd.DataFrame({
            "ticker": ["A", "B", "C", "D"],
            "pb": [-1.0, 1.0, 2.0, 3.0],
            "sector": ["Tech"] * 4,
        })
        # Shouldn't crash; negative values are just ranked last
        out = compute_sector_neutral_rank(df)
        assert len(out) == 4

    def test_no_duplicate_tickers_in_output(self):
        df = self._make_pb_df()
        out = compute_sector_neutral_rank(df)
        assert out["ticker"].duplicated().sum() == 0
