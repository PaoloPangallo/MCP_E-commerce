from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Iterator

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.agent.ebay_agent import EbayReactAgent
from app.agent.schemas import AgentRequest
from app.auth.dependencies import get_optional_user
from app.db.database import get_db

router = APIRouter()
logger = logging.getLogger(__name__)


def _normalize_llm_engine(llm_engine: str) -> str:
    llm_engine = (llm_engine or "").strip().lower()
    if llm_engine in {"gemini", "ollama", "rule_based"}:
        return llm_engine
    return "gemini"


def _sse(data: Dict[str, Any], event: str | None = None) -> str:
    payload = json.dumps(data, ensure_ascii=False)

    if event:
        return f"event: {event}\ndata: {payload}\n\n"

    return f"data: {payload}\n\n"


def _run_agent_stream_sync(
    query: str,
    llm_engine: str,
    db: Session,
    user: Any,
) -> Iterator[Dict[str, Any]]:
    """
    Wrapper sync isolato:
    l'agente esistente è sincrono, quindi lo teniamo qui.
    """
    agent = EbayReactAgent(db=db, user=user)

    request = AgentRequest(
        query=query,
        llm_engine=llm_engine,
        max_steps=4,
        return_trace=True,
    )

    for event in agent.run_stream(request):
        yield event


async def agent_event_generator(
    request: Request,
    query: str,
    llm_engine: str,
    db: Session,
    user: Any,
):
    llm_engine = _normalize_llm_engine(llm_engine)

    # apertura immediata stream
    yield _sse({
        "type": "start",
        "message": "agent stream started",
    })

    await asyncio.sleep(0)

    queue: asyncio.Queue[Dict[str, Any] | None] = asyncio.Queue()

    def worker():
        try:
            for event in _run_agent_stream_sync(query, llm_engine, db, user):
                asyncio.run_coroutine_threadsafe(queue.put(event), loop)
        except Exception as e:
            logger.exception("Agent stream failed: %s", e)
            asyncio.run_coroutine_threadsafe(
                queue.put({
                    "type": "error",
                    "message": str(e),
                }),
                loop,
            )
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    loop = asyncio.get_running_loop()
    worker_task = asyncio.to_thread(worker)
    background_task = asyncio.create_task(worker_task)

    try:
        while True:
            if await request.is_disconnected():
                logger.info("Client disconnected from /agent/stream")
                break

            try:
                event = await asyncio.wait_for(queue.get(), timeout=10.0)
            except asyncio.TimeoutError:
                # heartbeat per tenere vivo lo stream
                yield _sse({
                    "type": "heartbeat",
                    "message": "still running",
                })
                await asyncio.sleep(0)
                continue

            if event is None:
                break

            yield _sse(event)
            await asyncio.sleep(0)

        yield _sse({"type": "done"})

    except asyncio.CancelledError:
        logger.info("SSE stream cancelled by server/client")
        raise

    except Exception as e:
        logger.exception("SSE generator error: %s", e)
        yield _sse({
            "type": "error",
            "message": str(e),
        })

    finally:
        if not background_task.done():
            background_task.cancel()


@router.get("/agent/stream")
async def agent_stream(
    request: Request,
    query: str = Query(..., min_length=1),
    llm_engine: str = Query("gemini"),
    db: Session = Depends(get_db),
    user=Depends(get_optional_user),
):
    return StreamingResponse(
        agent_event_generator(request, query, llm_engine, db, user),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )