from fastapi import APIRouter, HTTPException

from app.services.feedback import get_seller_feedback
from app.services.trust import compute_trust_score
from app.services.nlp_sentiment import compute_sentiment_score


seller_router = APIRouter()


@seller_router.get("/seller/{seller_name}/feedback")
def get_feedback_route(seller_name: str):

    try:
        feedbacks = get_seller_feedback(seller_name, limit=10)

        sentiment_score = compute_sentiment_score(feedbacks)

        trust_score = compute_trust_score(
            feedbacks,
            sentiment_score=sentiment_score
        )

        return {
            "seller_name": seller_name,
            "count": len(feedbacks),
            "trust_score": trust_score,
            "sentiment_score": round(sentiment_score, 3),
            "feedbacks": feedbacks
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))