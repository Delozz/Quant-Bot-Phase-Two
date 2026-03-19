"""
Structured JSON logger for QuantBot.
"""
import logging
import json
import time
from pathlib import Path
from typing import Any


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        # Attach any extra fields passed via the `extra` kwarg
        for key, val in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and not key.startswith("_"):
                log_obj[key] = val
        return json.dumps(log_obj)


def get_logger(name: str, level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    """
    Returns a named logger with JSON console output and optional file output.

    Usage:
        logger = get_logger(__name__)
        logger.info("Processing ticker", extra={"ticker": "AAPL"})
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = JSONFormatter()

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class LatencyTimer:
    """Context manager that logs elapsed time of a code block."""

    def __init__(self, logger: logging.Logger, operation: str, **extra):
        self.logger = logger
        self.operation = operation
        self.extra = extra
        self._start: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        elapsed_ms = round((time.perf_counter() - self._start) * 1000, 2)
        self.logger.info(
            f"{self.operation} completed",
            extra={"operation": self.operation, "elapsed_ms": elapsed_ms, **self.extra},
        )
