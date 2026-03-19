"""
Data validation schemas for Phase 2 factor outputs.
Uses pandera for DataFrame-level validation.
"""
from __future__ import annotations

import pandera as pa
from pandera import Column, DataFrameSchema, Check


# ── Raw factor intermediate schemas ───────────────────────────────────────────

MeanReversionRawSchema = DataFrameSchema(
    {
        "ticker": Column(str, nullable=False),
        "mr_raw_z": Column(float, checks=[
            Check(lambda s: s.between(-5, 5), element_wise=False, error="mr_raw_z out of clip range"),
        ], nullable=True),
    },
    coerce=True,
)

SentimentRawSchema = DataFrameSchema(
    {
        "ticker": Column(str, nullable=False),
        "news_raw_velocity": Column(float, nullable=True),
    },
    coerce=True,
)

PBRawSchema = DataFrameSchema(
    {
        "ticker": Column(str, nullable=False),
        "pb_rank": Column(float, checks=[
            Check(lambda s: s.between(0, 1), element_wise=False, error="pb_rank must be in [0, 1]"),
        ], nullable=True),
    },
    coerce=True,
)


# ── Final factor output schema ─────────────────────────────────────────────────

FactorOutputSchema = DataFrameSchema(
    {
        "date": Column("datetime64[ns]", nullable=False),
        "ticker": Column(str, nullable=False),
        "mr_z": Column(float, nullable=True),
        "news_z": Column(float, nullable=True),
        "pb_z": Column(float, nullable=True),
    },
    coerce=True,
    checks=[
        Check(lambda df: df["ticker"].duplicated().sum() == 0, error="Duplicate tickers in output"),
    ],
)


def validate_factor_output(df) -> None:
    """
    Validate final factor DataFrame.
    Raises pandera.errors.SchemaError on failure.
    """
    FactorOutputSchema.validate(df, lazy=True)
