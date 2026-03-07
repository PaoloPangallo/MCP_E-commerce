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

router = APIRouter()
logger = logging.getLogger(__name__)

print("SEARCH ROUTER FILE:", os.path.abspath(__file__))


class SearchRequest(BaseModel):
    query: str
    llm_engine: Literal["gemini", "ollama", "rule_based"] = "ollama"


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/parse")
def parse(request: SearchRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query vuota")

    use_llm = request.llm_engine != "rule_based"

    try:
        return parse_query_service(
            request.query,
            use_llm=use_llm,
            include_meta=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore parser: {str(e)}")


@router.post("/search")
def search(
    request: SearchRequest,
    db: Session = Depends(get_db),
    user=Depends(get_optional_user),
):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query vuota")

    logger.info("Search query: %s", request.query)

    try:
        return run_search_pipeline(
            query=request.query,
            db=db,
            user=user,
            llm_engine=request.llm_engine,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Search pipeline error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agent")
def agent_search(
    request: AgentRequest,
    db: Session = Depends(get_db),
    user=Depends(get_optional_user),
):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query vuota")

    try:
        agent = EbayReactAgent(db=db, user=user)
        return agent.run(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Agent execution error")
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")