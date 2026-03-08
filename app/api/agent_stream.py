from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any, Dict, Iterator, Optional

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.agent.ebay_agent import EbayReactAgent
from app.agent.schemas import AgentRequest
from app.auth.dependencies import get_optional_user
from app.db.database import SessionLocal

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


def _validate_event(event: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(event, dict):
        return {
            "type": "error",
            "message": "Invalid event payload",
        }

    event_type = str(event.get("type") or "").strip().lower()
    allowed = {"start", "thinking", "tool_start", "tool_result", "final", "error", "heartbeat", "done"}

    if event_type not in allowed:
        return {
            "type": "error",
            "message": f"Unknown event type: {event_type or 'missing'}",
        }

    return event


def _run_agent_stream_sync(
    query: str,
    llm_engine: str,
    user: Any,
    stop_event: threading.Event,
) -> Iterator[Dict[str, Any]]:
    db: Optional[Session] = None

    try:
        db = SessionLocal()

        agent = EbayReactAgent(db=db, user=user)

        agent_request = AgentRequest(
            query=query,
            llm_engine=llm_engine,
            max_steps=6,
            return_trace=True,
        )

        for event in agent.run_stream(agent_request):
            if stop_event.is_set():
                logger.info("Stop event detected: interrompo stream agente")
                break
            yield _validate_event(event)

    finally:
        if db is not None:
            db.close()


async def agent_event_generator(
    request: Request,
    query: str,
    llm_engine: str,
    user: Any,
):
    llm_engine = _normalize_llm_engine(llm_engine)

    queue: asyncio.Queue[Dict[str, Any] | None] = asyncio.Queue()
    stop_event = threading.Event()
    loop = asyncio.get_running_loop()

    def worker():
        try:
            for event in _run_agent_stream_sync(query, llm_engine, user, stop_event):
                if stop_event.is_set():
                    break

                fut = asyncio.run_coroutine_threadsafe(queue.put(event), loop)
                try:
                    fut.result(timeout=2)
                except Exception:
                    logger.exception("Impossibile pushare evento nella coda SSE")
                    break

        except Exception as e:
            logger.exception("Agent stream failed: %s", e)
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    queue.put({
                        "type": "error",
                        "message": str(e),
                    }),
                    loop,
                )
                fut.result(timeout=2)
            except Exception:
                logger.exception("Impossibile pushare errore nella coda SSE")

        finally:
            try:
                fut = asyncio.run_coroutine_threadsafe(queue.put(None), loop)
                fut.result(timeout=2)
            except Exception:
                logger.exception("Impossibile chiudere la coda SSE")

    background_task = asyncio.create_task(asyncio.to_thread(worker))

    try:
        while True:
            if await request.is_disconnected():
                logger.info("Client disconnected from /agent/stream")
                stop_event.set()
                break

            try:
                event = await asyncio.wait_for(queue.get(), timeout=10.0)
            except asyncio.TimeoutError:
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

        if not await request.is_disconnected():
            yield _sse({"type": "done"})

    except asyncio.CancelledError:
        stop_event.set()
        logger.info("SSE stream cancelled by server/client")
        raise

    except Exception as e:
        stop_event.set()
        logger.exception("SSE generator error: %s", e)

        if not await request.is_disconnected():
            yield _sse({
                "type": "error",
                "message": str(e),
            })

    finally:
        stop_event.set()

        if not background_task.done():
            background_task.cancel()


@router.get("/agent/stream")
async def agent_stream(
    request: Request,
    query: str = Query(..., min_length=1),
    llm_engine: str = Query("gemini"),
    user=Depends(get_optional_user),
):
    return StreamingResponse(
        agent_event_generator(request, query, llm_engine, user),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )