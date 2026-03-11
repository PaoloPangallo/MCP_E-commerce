"""
Shared model singleton for SentenceTransformer.

All modules in the app should import their embedding model from here
to avoid loading the same model multiple times into memory.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_MODEL = None


def get_sentence_transformer():
    """Lazy-load and return the shared SentenceTransformer instance."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        logger.info("Loading shared SentenceTransformer: all-MiniLM-L6-v2")
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("SentenceTransformer loaded and ready.")
    return _MODEL


def preload():
    """Call this at app startup to warm up the model on the main thread."""
    get_sentence_transformer()
