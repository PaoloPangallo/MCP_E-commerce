from typing import List, Dict
from datetime import datetime
from math import exp


# ============================================================
# FEEDBACK RATIO
# ============================================================

def _feedback_ratio(feedbacks: List[Dict]) -> float:

    total = len(feedbacks)

    if total == 0:
        return 0.5

    score = 0

    for f in feedbacks:

        r = f.get("rating")

        if r == 5: # Positive
            score += 1

        elif r == 3: # Neutral
            score += 0.5

        elif r == 1: # Negative
            score += 0

    return score / total


# ============================================================
# RECENCY SCORE
# ============================================================

def _recency_score(feedbacks: List[Dict]) -> float:

    if not feedbacks:
        return 0.5

    now = datetime.utcnow()

    weights = []
    values = []

    for f in feedbacks:

        date = f.get("time")

        if not date:
            continue

        try:

            if isinstance(date, str):
                date = datetime.fromisoformat(date.replace("Z", ""))

            age_days = (now - date).days

            weight = exp(-age_days / 365)

            r = f.get("rating")

            if r == 5: # Positive
                val = 1
            elif r == 3: # Neutral
                val = 0.5
            else: # Negative (1)
                val = 0

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

    # logistic confidence curve
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
        sentiment_score = ratio

    trust_score = (
        0.45 * ratio +
        0.25 * sentiment_score +
        0.20 * recency +
        0.10 * confidence
    )

    return round(trust_score, 3)