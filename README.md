# QuantBot — Phase 2: Factor Component Engine

> Owner: Devon Lopez

Computes three normalized Z-score factor signals per stock in the S&P 500 universe daily.

## Factors

| # | Factor | Signal | Trigger |
|---|--------|--------|---------|
| 1 | **Mean Reversion** | Z-score of PCA residual vs 20-day rolling stats | `mr_z ≤ -2.0` |
| 2 | **Sentiment Velocity** | FinnBERT 24h sentiment − 7d sentiment (decay-weighted) | Positive velocity |
| 3 | **Price-to-Book** | Sector-neutral percentile rank (inverted) | Bottom 20% in sector |

All factors are winsorized (1% each tail) then cross-sectionally Z-scored.

---

## Output Schema

```
date     : datetime64  — trading date
ticker   : str         — stock symbol
mr_z     : float       — mean reversion z-score   (negative = oversold)
news_z   : float       — sentiment velocity z-score (positive = improving)
pb_z     : float       — value z-score             (positive = cheap in sector)
```

Saved to: `data/features/factors/date=YYYY-MM-DD/factors.parquet`

---

## Quick Start

### 1. Install dependencies

```bash
# With Poetry (recommended)
poetry install

# Or with pip
pip install numpy pandas scipy scikit-learn statsmodels transformers torch \
            aiohttp httpx tenacity duckdb pyarrow pydantic pandera pyyaml yfinance
```

### 2. Configure API keys

```bash
cp config/api_keys.yaml.template config/api_keys.yaml
# Edit config/api_keys.yaml with your LSEG credentials
```

### 3. Run the pipeline

```bash
# Today's factors
python main.py

# Specific date
python main.py --date 2026-03-10

# Scheduled daemon (fires at 18:30 EST daily)
python main.py --schedule

# Load saved factors
python main.py --load --start 2026-03-01 --end 2026-03-14
```

### 4. Run tests

```bash
python main.py --test
# or directly:
pytest tests/ -v
```

---

## Architecture

```
main.py
└── src/pipeline/factor_pipeline.py   ← generate_factor_scores(date)
    ├── Factor 1: mean_reversion/
    │   ├── residual_loader.py         ← loads Phase 1 PCA residuals
    │   ├── rolling_stats.py           ← 20-day rolling mean/std
    │   └── mean_reversion_signal.py   ← z-score + clip
    │
    ├── Factor 2: sentiment_velocity/
    │   ├── finnbert_classifier.py     ← ProsusAI/finbert batch inference
    │   ├── sentiment_aggregator.py    ← decay-weighted 24h / 7d windows
    │   └── sentiment_velocity.py     ← velocity = s24h − s7d
    │
    ├── Factor 3: valuation/
    │   ├── pb_ratio_fetcher.py        ← fetch P/B from LSEG / yfinance
    │   ├── sector_ranker.py           ← sector-neutral percentile rank
    │   └── value_score.py             ← final pb_rank signal
    │
    ├── normalization/
    │   ├── winsorize.py               ← 1% tail clipping
    │   └── zscore.py                  ← cross-sectional z-score
    │
    └── ingestion/
        ├── lseg_client.py             ← LSEG API + yfinance fallback
        ├── news_collector.py          ← async news fetch + cleaning
        └── fundamentals_collector.py  ← P/B + sector data
```

---

## Notes

- **Phase 1 dependency**: Place PCA residuals at `data/processed/pca_residuals.parquet`.
  The system generates synthetic residuals for development if the file is missing.
- **LSEG fallback**: If LSEG credentials are not configured, price/fundamental data
  falls back to `yfinance` automatically. News sentiment uses empty signals.
- **FinnBERT**: Downloaded automatically from HuggingFace on first run (~400MB).
  Uses GPU if available, otherwise CPU.
- **Phase 3 integration**: Import `generate_factor_scores` directly:
  ```python
  from src.pipeline.factor_pipeline import generate_factor_scores
  df = generate_factor_scores(date)  # → DataFrame(ticker, mr_z, news_z, pb_z)
  ```
# Quant-Bot-Phase-Two
