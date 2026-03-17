import logging
import os
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.agent.ebay_agent import EbayReactAgent
from app.agent.schemas import AgentRequest
from app.auth.dependencies import get_optional_user
from app.db.database import get_db
from app.services.parser import parse_query_service
from app.services.search_pipeline import run_search_pipeline
from app.services.memory_service import clear_session_memory

router = APIRouter()
logger = logging.getLogger(__name__)

_IS_PROD = os.getenv("ENV", "development").strip().lower() in {"production", "prod"}


class SearchRequest(BaseModel):
    query: str
    llm_engine: Literal["gemini", "ollama", "rule_based"] = "gemini"


@router.post("/parse")
async def parse(request: SearchRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query vuota")

    use_llm = request.llm_engine != "rule_based"

    try:
        return await parse_query_service(
            request.query,
            use_llm=use_llm,
            include_meta=True,
        )
    except Exception as e:
        logger.exception("Parse error")
        raise HTTPException(
            status_code=500,
            detail="Errore interno del server." if _IS_PROD else f"Errore parser: {str(e)}"
        )


@router.post("/search")
async def search(
    request: SearchRequest,
    db: Session = Depends(get_db),
    user=Depends(get_optional_user),
):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query vuota")

    logger.info("Search query: %s", request.query)

    try:
        return await run_search_pipeline(
            query=request.query,
            db=db,
            user=user,
            llm_engine=request.llm_engine,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Search pipeline error")
        raise HTTPException(
            status_code=500,
            detail="Errore interno del server." if _IS_PROD else str(e)
        )


@router.post("/agent")
async def agent_search(
    request: AgentRequest,
    db: Session = Depends(get_db),
    user=Depends(get_optional_user),
):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query vuota")

    try:
        agent = EbayReactAgent(db=db, user=user)
        result = await agent.run(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Agent execution error")
        raise HTTPException(
            status_code=500,
            detail="Errore interno del server." if _IS_PROD else f"Agent error: {str(e)}"
        )


@router.delete("/agent/memory")
async def wipe_agent_memory(
    user=Depends(get_optional_user)
):
    """
    Clears the Redis session memory for the current user.
    If anonymous, we clear the memory of session ID '1' (default guest).
    """
    user_key = str(user.id) if user and hasattr(user, "id") else "1"
    try:
        clear_session_memory(user_key)
        return {"status": "ok", "message": "Memoria di sessione cancellata. L'agente eBay ora riparte da una context window pulita."}
    except Exception as e:
        logger.exception("Error clearing memory")
        raise HTTPException(status_code=500, detail=f"Errore pulizia memoria: {str(e)}")