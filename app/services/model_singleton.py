"""
Shared model singleton for SentenceTransformer.

All modules in the app should import their embedding model from here
to avoid loading the same model multiple times into memory.
"""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

_MODEL = None
_MODEL_LOCK = threading.Lock()


def get_sentence_transformer():
    """Lazy-load and return the shared SentenceTransformer instance (thread-safe)."""
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                from sentence_transformers import SentenceTransformer  # noqa: PLC0415
                logger.info("Loading shared SentenceTransformer: all-MiniLM-L6-v2")
                _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("SentenceTransformer loaded and ready.")
    return _MODEL


_SENTIMENT_PIPELINE = None
_SENTIMENT_LOCK = threading.Lock()

def get_sentiment_pipeline():
    """Lazy-load and return the specialized Multilingual Sentiment pipeline."""
    global _SENTIMENT_PIPELINE
    if _SENTIMENT_PIPELINE is None:
        with _SENTIMENT_LOCK:
            if _SENTIMENT_PIPELINE is None:
                from transformers import pipeline  # noqa: PLC0415
                logger.info("Loading dedicated NLP Sentiment pipeline: nlptown/bert-base-multilingual-uncased-sentiment")
                # Warning: downloads ~600MB on first run
                _SENTIMENT_PIPELINE = pipeline(
                    "sentiment-analysis",
                    model="nlptown/bert-base-multilingual-uncased-sentiment",
                    device=-1 # CPU by default, switch if GPU auto-detect needed
                )
                logger.info("Sentiment pipeline loaded and ready.")
    return _SENTIMENT_PIPELINE


def preload():
    """Call this at app startup to warm up the model on the main thread."""
    get_sentence_transformer()
    # Non pre-carichiamo il sentiment model a meno che non serva per evitare picchi di RAM al boot se non utilizzato.
