from fastapi import APIRouter, HTTPException, Query

from app.config.settings import settings
from app.services.seller_pipeline import run_seller_pipeline

seller_router = APIRouter()


@seller_router.get("/seller/{seller_name}/feedback")
async def get_feedback_route(
    seller_name: str,
    page: int = Query(1, ge=1, le=100),
    limit: int = Query(50, ge=1, le=200),
):
    try:
        return await run_seller_pipeline(
            seller_name=seller_name,
            page=page,
            limit=limit,
        )
    except Exception as e:
        detail = "Errore interno del server." if settings.is_production else str(e)
        raise HTTPException(status_code=500, detail=detail)