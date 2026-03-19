"""
Two-tier caching layer:
  1. Local disk cache  (always available)
  2. Redis cache       (optional; falls back to disk silently)

Usage:
    cache = Cache(prefix="news")
    cache.set("AAPL:2026-03-10", articles_list, ttl=3600)
    data = cache.get("AAPL:2026-03-10")
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

_DISK_CACHE_DIR = Path("data/.cache")


def _disk_path(key: str) -> Path:
    safe = hashlib.sha256(key.encode()).hexdigest()
    return _DISK_CACHE_DIR / safe[:2] / (safe + ".pkl")


class Cache:
    def __init__(self, prefix: str = "", redis_url: str | None = None):
        self.prefix = prefix
        self._redis = None
        if redis_url:
            try:
                import redis

                self._redis = redis.from_url(redis_url, decode_responses=False)
                self._redis.ping()
                logger.info("Redis cache connected", extra={"url": redis_url})
            except Exception as exc:
                logger.warning("Redis unavailable, falling back to disk cache", extra={"error": str(exc)})
                self._redis = None

    def _full_key(self, key: str) -> str:
        return f"{self.prefix}:{key}" if self.prefix else key

    # ------------------------------------------------------------------
    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        fk = self._full_key(key)
        payload = pickle.dumps(value)
        if self._redis:
            try:
                self._redis.set(fk, payload, ex=ttl)
                return
            except Exception as exc:
                logger.warning("Redis set failed", extra={"key": fk, "error": str(exc)})
        # Fallback: disk
        path = _disk_path(fk)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    def get(self, key: str) -> Any | None:
        fk = self._full_key(key)
        if self._redis:
            try:
                raw = self._redis.get(fk)
                if raw:
                    return pickle.loads(raw)
            except Exception as exc:
                logger.warning("Redis get failed", extra={"key": fk, "error": str(exc)})
        # Fallback: disk
        path = _disk_path(fk)
        if path.exists():
            return pickle.loads(path.read_bytes())
        return None

    def delete(self, key: str) -> None:
        fk = self._full_key(key)
        if self._redis:
            try:
                self._redis.delete(fk)
            except Exception:
                pass
        path = _disk_path(fk)
        if path.exists():
            path.unlink()

    def exists(self, key: str) -> bool:
        return self.get(key) is not None
