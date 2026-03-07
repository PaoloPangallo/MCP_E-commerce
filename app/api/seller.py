from fastapi import APIRouter, HTTPException

from app.services.seller_pipeline import run_seller_pipeline

seller_router = APIRouter()


@seller_router.get("/seller/{seller_name}/feedback")
def get_feedback_route(
    seller_name: str,
    page: int = 1,
    limit: int = 10
):
    try:
        return run_seller_pipeline(
            seller_name=seller_name,
            page=page,
            limit=limit,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))