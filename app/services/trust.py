from typing import List, Dict
from datetime import datetime
from math import exp


def _feedback_ratio(feedbacks: List[Dict]) -> float:
    """
    Basic positive/negative ratio.
    """
    total = len(feedbacks)

    if total == 0:
        return 0.5

    pos = sum(1 for f in feedbacks if f.get("rating") == "Positive")
    neg = sum(1 for f in feedbacks if f.get("rating") == "Negative")

    score = (pos - neg) / total

    # normalize -1..1 → 0..1
    return (score + 1) / 2


def _recency_score(feedbacks: List[Dict]) -> float:
    """
    Weight recent reviews more.
    """
    if not feedbacks:
        return 0.5

    now = datetime.utcnow()

    weights = []
    values = []

    for f in feedbacks:
        date = f.get("date")

        if not date:
            continue

        try:
            if isinstance(date, str):
                date = datetime.fromisoformat(date)

            age_days = (now - date).days

            weight = exp(-age_days / 365)  # decay over 1 year

            val = 1 if f.get("rating") == "Positive" else 0

            weights.append(weight)
            values.append(val * weight)

        except Exception:
            continue

    if not weights:
        return 0.5

    return sum(values) / sum(weights)


def compute_trust_score(
    feedbacks: List[Dict],
    sentiment_score: float | None = None
) -> float:

    if not feedbacks:
        return 0.5

    ratio = _feedback_ratio(feedbacks)
    recency = _recency_score(feedbacks)

    # default sentiment neutral
    if sentiment_score is None:
        sentiment_score = ratio

    trust_score = (
        0.5 * ratio +
        0.3 * sentiment_score +
        0.2 * recency
    )

    return round(trust_score, 3)