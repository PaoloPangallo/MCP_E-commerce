from __future__ import annotations

import hashlib
import logging
from math import exp
from typing import Dict, List, Sequence, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# MODEL (lazy singleton)
# ============================================================

_POS_WORDS = {"excellent", "great", "perfect", "good", "fast", "ottimo", "perfetto", "veloce", "consigliato", "super", "ok"}
_NEG_WORDS = {"terrible", "fake", "bad", "pessimo", "truffa", "rotto", "lento", "sconsigliato", "scam"}


from app.services.model_singleton import get_sentiment_pipeline
from app.llm.client import call_llm


# ============================================================
# TEXT EXTRACTION
# ============================================================

def _extract_texts(feedbacks: List[Dict], max_texts: int = 20) -> List[str]:
    texts: List[str] = []

    for f in feedbacks:
        text = (
            f.get("CommentText")
            or f.get("comment")
            or f.get("text")
        )

        if not isinstance(text, str):
            continue

        cleaned = text.strip()
        if len(cleaned) < 4:
            continue

        texts.append(cleaned)

        if len(texts) >= max_texts:
            break

    return texts


def _stable_text_fingerprint(texts: Sequence[str]) -> str:
    joined = "||".join(sorted(t.strip().lower() for t in texts if t.strip()))
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


# ============================================================
# EMBEDDING ANCHORS
# ============================================================

# _anchor_vectors removed as we now use a dedicated pipeline


def _heuristic_sentiment_single(text: str) -> float:
    t = text.lower()
    total_pos = sum(1 for w in _POS_WORDS if w in t)
    total_neg = sum(1 for w in _NEG_WORDS if w in t)

    total_hits = total_pos + total_neg
    if total_hits == 0:
        return 0.5

    ratio = (total_pos - total_neg) / (total_hits + 1)
    return round(float(1 / (1 + exp(-3 * ratio))), 4)


# ============================================================
# CACHE SINGOLA STRINGA
# ============================================================

_TEXT_SENTIMENT_CACHE: Dict[str, float] = {}
_TEXT_CACHE_MAX = 5000

def _cache_put_text(text: str, score: float) -> None:
    if len(_TEXT_SENTIMENT_CACHE) >= _TEXT_CACHE_MAX:
        # Evict earliest 10%
        keys_to_delete = list(_TEXT_SENTIMENT_CACHE.keys())[: (_TEXT_CACHE_MAX // 10)]
        for k in keys_to_delete:
            del _TEXT_SENTIMENT_CACHE[k]
    _TEXT_SENTIMENT_CACHE[text] = score


# ============================================================
# PUBLIC API: SINGLE SCORES INJECTION
# ============================================================

def compute_sentiment_score(
    feedbacks: List[Dict],
    max_texts: int = 150,
    use_cache: bool = True,
    use_fast_path: bool = False,
) -> float:
    """
    Inietta la proprietà `nlp_sentiment` [0,1] all'interno di ogni dict in `feedbacks`
    fino a `max_texts`. Restituisce la media aritmetica dei punteggi.
    """
    if not feedbacks:
        return 0.5

    texts_to_encode = []
    feedbacks_to_encode = []

    for f in feedbacks:
        text = str(f.get("CommentText") or f.get("comment") or f.get("text") or "").strip()
        if len(text) < 4:
            f["nlp_sentiment"] = 0.5
            continue

        if use_cache and text in _TEXT_SENTIMENT_CACHE:
            f["nlp_sentiment"] = _TEXT_SENTIMENT_CACHE[text]
        else:
            texts_to_encode.append(text)
            feedbacks_to_encode.append(f)

        if len(texts_to_encode) >= max_texts:
            break

    # Riempiamo anche i feedback restanti non toccati, se sono oltre max_texts
    # in modo da non avere None accidentali
    for f in feedbacks:
        if "nlp_sentiment" not in f:
            f["nlp_sentiment"] = 0.5

    # Elaboriamo quelli nuovi
    if texts_to_encode:
        if use_fast_path and len(texts_to_encode) <= 8:
            for text, f in zip(texts_to_encode, feedbacks_to_encode):
                val = _heuristic_sentiment_single(text)
                f["nlp_sentiment"] = val
                if use_cache:
                    _cache_put_text(text, val)
        else:
            try:
                pipeline = get_sentiment_pipeline()
                
                # Execute pipeline on the batch
                results = pipeline(texts_to_encode)
                
                for text, f, res in zip(texts_to_encode, feedbacks_to_encode, results):
                    label = res.get('label', '3 stars')
                    try:
                        stars = int(label.split()[0])
                    except (ValueError, IndexError):
                        stars = 3

                    # Map 1-5 stars to 0.0-1.0 float score
                    if stars == 5: score = 0.95
                    elif stars == 4: score = 0.75
                    elif stars == 3: score = 0.50
                    elif stars == 2: score = 0.25
                    else: score = 0.05
                    
                    f["nlp_sentiment"] = score
                    if use_cache:
                        _cache_put_text(text, score)

            except Exception as e:
                logger.warning("Embedding sentiment failed, using heuristic: %s", e)
                for text, f in zip(texts_to_encode, feedbacks_to_encode):
                    val = _heuristic_sentiment_single(text)
                    f["nlp_sentiment"] = val
                    if use_cache:
                        _cache_put_text(text, val)

    # Calcoliamo l'overall
    valid_scores = []
    for f in feedbacks:
        base_nlp = f.get("nlp_sentiment", 0.5)
        # Se presente un rating eBay effettivo (1 neg, 3 neu, 5 pos), si bilancia 
        # l'NLP con il rating oggettivo in modo che 99 recensioni a 5 stelle non passino per neutre.
        rating = f.get("rating", 3)
        if rating >= 4:
            explicit_val = 0.90
        elif rating <= 2:
            explicit_val = 0.10
        else:
            explicit_val = 0.50
        
        # Blend: 60% star rating, 40% text NLP analysis
        blended = (explicit_val * 0.6) + (base_nlp * 0.4)
        f["nlp_sentiment"] = round(blended, 4)
        valid_scores.append(blended)

    return round(float(np.mean(valid_scores)), 4) if valid_scores else 0.5

# ============================================================
# LLM SENTIMENT LABEL FOR RAG INGESTION
# ============================================================

async def _call_llm(prompt: str) -> Optional[str]:
    try:
        result, _ = await call_llm(prompt)
        return result
    except Exception as exc:
        logger.warning("Sentiment LLM failed: %s", exc)
    return None

async def extract_sentiment_label(text: str) -> str:
    """
    Uses LLM (fastest available, e.g. Gemini) to label a single feedback 
    as exactly one word: POSITIVE, NEGATIVE, or NEUTRAL.
    """
    if not text.strip():
        return "NEUTRAL"
        
    prompt = f"""
    Analyze the following seller feedback comment and classify its sentiment.
    Return ONLY ONE WORD from this list: POSITIVE, NEGATIVE, NEUTRAL.
    No other text, no punctuation.

    Comment: "{text}"
    """
    try:
        # Fallback to Ollama if Gemini is not set/fails, but call_gemini is usually faster
        res = await _call_llm(prompt)
        if res:
            res = res.strip().upper()
            if "POSITIVE" in res: return "POSITIVE"
            if "NEGATIVE" in res: return "NEGATIVE"
        return "NEUTRAL"
    except Exception as e:
        logger.warning("extract_sentiment_label LLM failed: %s", e)
        # fallback to heuristic
        pos_hits = sum(1 for w in _POS_WORDS if w in text.lower())
        neg_hits = sum(1 for w in _NEG_WORDS if w in text.lower())
        if pos_hits > neg_hits: return "POSITIVE"
        if neg_hits > pos_hits: return "NEGATIVE"
        return "NEUTRAL"

