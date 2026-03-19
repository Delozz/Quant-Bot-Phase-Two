"""
Loads settings.yaml and api_keys.yaml into typed config objects.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


# ── Pydantic models ────────────────────────────────────────────────────────────

class MeanReversionConfig(BaseModel):
    rolling_window: int = 20
    z_trigger: float = -2.0
    clip_range: list[float] = [-5.0, 5.0]

class SentimentConfig(BaseModel):
    short_window_hours: int = 24
    long_window_days: int = 7
    min_articles: int = 5
    min_relevance: float = 0.7
    batch_size: int = 32
    decay_lambda: float = 0.1

class PBConfig(BaseModel):
    bottom_pct_threshold: float = 0.20

class FactorsConfig(BaseModel):
    mean_reversion: MeanReversionConfig = Field(default_factory=MeanReversionConfig)
    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)
    pb_ratio: PBConfig = Field(default_factory=PBConfig)

class NormalizationConfig(BaseModel):
    winsorize_limits: list[float] = [0.01, 0.01]

class StorageConfig(BaseModel):
    factors_path: str = "data/features/factors.parquet"
    raw_news_path: str = "data/raw/news"
    raw_fundamentals_path: str = "data/raw/fundamentals"
    residuals_path: str = "data/processed/pca_residuals.parquet"

class UniverseConfig(BaseModel):
    index: str = "S&P500"
    min_avg_volume: int = 1_000_000

class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"
    output: str = "logs/quantbot.log"

class Settings(BaseModel):
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    factors: FactorsConfig = Field(default_factory=FactorsConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

class LSEGKeys(BaseModel):
    app_key: str = ""
    username: str = ""
    password: str = ""

class RedisKeys(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None

class APIKeys(BaseModel):
    lseg: LSEGKeys = Field(default_factory=LSEGKeys)
    redis: RedisKeys = Field(default_factory=RedisKeys)


# ── Loaders ────────────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data = _load_yaml(Path("config/settings.yaml"))
    return Settings(**data)


@lru_cache(maxsize=1)
def get_api_keys() -> APIKeys:
    data = _load_yaml(Path("config/api_keys.yaml"))
    return APIKeys(**data)
