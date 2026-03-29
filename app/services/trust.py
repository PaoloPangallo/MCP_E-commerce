import logging
from typing import List, Dict
from datetime import datetime, timezone
from math import exp

logger = logging.getLogger(__name__)

# Linear mapping: any numeric rating → [0, 1]
# eBay uses 1 (Negative), 3 (Neutral), 5 (Positive).
# Ratings 2/4 or unknown map linearly so no value silently becomes 0.
_MAX_RATING = 5.0


def _rating_to_value(r) -> float | None:
    """Convert a numeric rating to [0, 1]. Returns None if unusable."""
    try:
        rv = float(r)
        if rv <= 0 or rv > _MAX_RATING:
            return None
        return (rv - 1) / (_MAX_RATING - 1)
    except (TypeError, ValueError):
        return None


# ============================================================
# FEEDBACK RATIO
# ============================================================

def _feedback_ratio(feedbacks: List[Dict]) -> float:
    total = len(feedbacks)
    if total == 0:
        return 0.5

    score = 0.0
    counted = 0
    for f in feedbacks:
        val = _rating_to_value(f.get("rating"))
        if val is not None:
            score += val
            counted += 1

    return score / counted if counted else 0.5


# ============================================================
# RECENCY SCORE
# ============================================================

def _recency_score(feedbacks: List[Dict]) -> float:
    if not feedbacks:
        return 0.5

    now = datetime.now(timezone.utc)
    weights: List[float] = []
    values: List[float] = []

    for f in feedbacks:
        date = f.get("time")
        if not date:
            continue
        try:
            if isinstance(date, str):
                date_parsed = datetime.fromisoformat(date.replace("Z", "+00:00"))
            else:
                date_parsed = date
            # Make sure both are timezone-aware for comparison
            if date_parsed.tzinfo is None:
                date_parsed = date_parsed.replace(tzinfo=timezone.utc)

            age_days = max(0, (now - date_parsed).days)
            weight = exp(-age_days / 365)

            val = _rating_to_value(f.get("rating"))
            if val is None:
                continue

            weights.append(weight)
            values.append(val * weight)
        except Exception:
            continue

    if not weights:
        return 0.5

    return sum(values) / sum(weights)


# ============================================================
# CONFIDENCE SCORE
# ============================================================

def _confidence(feedbacks: List[Dict]) -> float:
    n = len(feedbacks)
    return 1 - exp(-n / 50)


# ============================================================
# TRUST SCORE
# ============================================================

def compute_trust_score(
    feedbacks: List[Dict],
    sentiment_score: float | None = None
) -> float:

    if not feedbacks:
        return 0.5

    ratio = _feedback_ratio(feedbacks)
    recency = _recency_score(feedbacks)
    confidence = _confidence(feedbacks)

    if sentiment_score is None:
        logger.debug(
            "compute_trust_score: sentiment_score not provided, falling back to ratio (%.3f). "
            "Sentiment weight will be absorbed by ratio.",
            ratio,
        )
        sentiment_score = ratio

    trust_score = (
        0.45 * ratio +
        0.25 * sentiment_score +
        0.20 * recency +
        0.10 * confidence
    )

    return round(trust_score, 3)
