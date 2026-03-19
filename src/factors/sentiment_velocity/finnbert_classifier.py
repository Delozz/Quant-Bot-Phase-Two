"""
FinnBERT sentiment classifier.

Model:  ProsusAI/finbert
Labels: positive (+1), neutral (0), negative (-1)

Runs inference in batches for throughput efficiency.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from src.utils.config import get_settings
from src.utils.logger import get_logger, LatencyTimer

logger = get_logger(__name__)

# ── Scored output ──────────────────────────────────────────────────────────────

@dataclass
class SentimentResult:
    text: str
    label: str         # "positive" | "neutral" | "negative"
    score: float       # model confidence [0, 1]
    numeric: float     # +1 / 0 / -1

_LABEL_MAP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


# ── Classifier singleton ───────────────────────────────────────────────────────

_tokenizer = None
_model = None
_device: str = "cpu"


def _load_model() -> None:
    global _tokenizer, _model, _device
    if _tokenizer is not None:
        return

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    MODEL_NAME = "ProsusAI/finbert"
    logger.info("Loading FinnBERT model", extra={"model": MODEL_NAME})
    with LatencyTimer(logger, "finbert_model_load"):
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = _model.to(_device)
        _model.eval()
    logger.info("FinnBERT model loaded", extra={"device": _device})


# ── Batch inference ────────────────────────────────────────────────────────────

def classify_batch(texts: Sequence[str]) -> list[SentimentResult]:
    """
    Classify a list of texts. Returns one SentimentResult per input.
    Handles empty input gracefully.
    """
    if not texts:
        return []

    _load_model()
    cfg = get_settings().factors.sentiment
    batch_size = cfg.batch_size

    results: list[SentimentResult] = []
    n_batches = math.ceil(len(texts) / batch_size)

    with LatencyTimer(logger, "finbert_inference", texts=len(texts), batches=n_batches):
        for i in range(0, len(texts), batch_size):
            batch_texts = list(texts[i : i + batch_size])
            inputs = _tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(_device)

            with torch.no_grad():
                outputs = _model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            label_names = [_model.config.id2label[j] for j in range(probs.shape[-1])]
            for text, prob_row in zip(batch_texts, probs):
                idx = int(prob_row.argmax())
                label = label_names[idx].lower()
                results.append(
                    SentimentResult(
                        text=text,
                        label=label,
                        score=float(prob_row[idx]),
                        numeric=_LABEL_MAP.get(label, 0.0),
                    )
                )
    return results
