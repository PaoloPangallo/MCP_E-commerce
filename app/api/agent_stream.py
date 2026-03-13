from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.agent.ebay_agent import EbayReactAgent
from app.agent.schemas import AgentRequest
from app.auth.dependencies import get_optional_user
from app.db.database import SessionLocal

router = APIRouter()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

MAX_QUERY_LENGTH = 500

# Request concurrency limit to prevent starving the server
MAX_CONCURRENT_STREAMS = int(os.getenv("MAX_CONCURRENT_STREAMS", "10"))
_stream_semaphore = asyncio.Semaphore(MAX_CONCURRENT_STREAMS)

WORKER_HARD_TIMEOUT_SECONDS = 180.0
QUEUE_WAIT_TIMEOUT_SECONDS = 75.0
HEARTBEAT_INTERVAL_SECONDS = 10.0 # Modified: Changed from 15.0 to 10.0

_ALLOWED_EVENT_TYPES = {
    "start",
    "thinking",
    "tool_start",
    "tool_result",
    "final",
    "error",
    "heartbeat",
    "done",
}

_SSE_PREFIX_RE = re.compile(r"^\s*data\s*:\s*\{", re.IGNORECASE)
_EVENT_STREAM_MARKERS = (
    '"type": "start"',
    '"type":"start"',
    '"type": "thinking"',
    '"type":"thinking"',
    '"type": "tool_start"',
    '"type":"tool_start"',
    '"type": "tool_result"',
    '"type":"tool_result"',
    '"type": "final"',
    '"type":"final"',
    '"type": "done"',
    '"type":"done"',
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _normalize_llm_engine(llm_engine: str) -> str:
    llm_engine = (llm_engine or "").strip().lower()
    if llm_engine in {"gemini", "ollama", "rule_based"}:
        return llm_engine
    return "ollama"


def _sanitize_query(query: str) -> str:
    q = str(query or "").strip()
    if not q:
        return ""

    lowered = q.lower()

    if _SSE_PREFIX_RE.search(q):
        return ""

    if "data:" in lowered and any(marker in lowered for marker in _EVENT_STREAM_MARKERS):
        return ""

    # Prompt injection guardrails: strip system/instruction overrides
    injection_patterns = [
        r"<\|.*?\|>",  # Token markers
        r"(?i)\bignore\s+(all\s+)?(previous\s+)?(instructions|directions)\b",
        r"(?i)\bsystem(\s+prompt)?\s*:",
        r"(?i)\bnew\s+instructions\s*:",
        r"(?i)\byou\s+are\s+now\b"
    ]
    for pattern in injection_patterns:
        q = re.sub(pattern, "", q)

    q = re.sub(r"\s+", " ", q).strip()

    if len(q) > MAX_QUERY_LENGTH:
        q = q[:MAX_QUERY_LENGTH].rstrip()

    return q


def _looks_like_event_stream_payload(query: str) -> bool:
    q = str(query or "").strip().lower()
    if not q:
        return False

    if _SSE_PREFIX_RE.search(q):
        return True

    if "data:" in q and any(marker in q for marker in _EVENT_STREAM_MARKERS):
        return True

    return False


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
    if event_type not in _ALLOWED_EVENT_TYPES:
        return {
            "type": "error",
            "message": f"Unknown event type: {event_type or 'missing'}",
        }

    if event_type == "tool_result":
        backend = event.get("backend")
        tool = event.get("tool")
        logger.info(
            "Validated tool_result event | tool=%s | backend=%s | ok=%s | status=%s",
            tool,
            backend,
            event.get("ok"),
            event.get("status"),
        )

    return event


async def agent_event_generator(
    request: Request,
    query: str,
    llm_engine: str,
    user: Any,
):
    llm_engine = _normalize_llm_engine(llm_engine)
    query = _sanitize_query(query)

    if not query:
        yield _sse(
            {
                "type": "error",
                "message": "Query non valida o vuota.",
            }
        )
        yield _sse({"type": "done"})
        return

    db: Optional[Session] = None
    done_sent = False
    last_heartbeat = time.monotonic()

    try:
        db = SessionLocal()

        agent = EbayReactAgent(db=db, user=user)

        logger.info(
            "Agent created | mcp_server_url=%s | prefer_mcp=%s | strict_mcp=%s",
            getattr(agent, "mcp_server_url", None),
            getattr(agent, "prefer_mcp", None),
            getattr(agent, "strict_mcp", None),
        )

        agent_request = AgentRequest(
            query=query,
            llm_engine=llm_engine,
            max_steps=6,
            return_trace=True,
        )

        logger.info(
            "Starting async agent stream | query=%s | llm_engine=%s | timeout=%ss",
            query,
            llm_engine,
            WORKER_HARD_TIMEOUT_SECONDS
        )

        start_time = time.monotonic()
        async with _stream_semaphore:
            async for event in agent.run_stream(agent_request):
                if time.monotonic() - start_time > WORKER_HARD_TIMEOUT_SECONDS:
                    logger.warning("Agent stream hit hard timeout of %ss", WORKER_HARD_TIMEOUT_SECONDS)
                    yield _sse({
                        "type": "error",
                        "message": "La richiesta ha richiesto troppo tempo ed è andata in timeout."
                    })
                    break

                if await request.is_disconnected():
                    logger.info("Client disconnected from /agent/stream")
                    break

                validated = _validate_event(event)
                event_type = validated.get("type", "")

                yield _sse(validated)

                if event_type == "done":
                    done_sent = True

                # Send heartbeats during gaps
                now = time.monotonic()
                if now - last_heartbeat >= HEARTBEAT_INTERVAL_SECONDS:
                    yield _sse({"type": "heartbeat"})
                    last_heartbeat = now
                    
                await asyncio.sleep(0)  # Yield control to event loop

        if not done_sent and not await request.is_disconnected():
            done_sent = True
            yield _sse({"type": "done"})

    except asyncio.CancelledError:
        logger.info("SSE stream cancelled by server/client")
        # Do NOT re-raise inside an async generator — ASGI handles the cleanup
        # Raising here causes "ASGI callable returned without completing response"

    except Exception as exc:
        logger.exception("SSE generator error: %s", exc)

        if not await request.is_disconnected():
            yield _sse(
                {
                    "type": "error",
                    "message": str(exc),
                }
            )

            if not done_sent:
                done_sent = True
                yield _sse({"type": "done"})

    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                logger.warning("Failed closing DB session in agent stream")


@router.get("/agent/stream")
async def agent_stream(
    request: Request,
    query: str = Query(..., min_length=1),
    llm_engine: str = Query("ollama"),
    user=Depends(get_optional_user),
):
    clean_query = _sanitize_query(query)

    if not clean_query:
        raise HTTPException(status_code=400, detail="Query non valida o vuota.")

    if _looks_like_event_stream_payload(query):
        raise HTTPException(
            status_code=400,
            detail="La query sembra un payload SSE/event stream, non un testo utente.",
        )

    logger.info(
        "Incoming /agent/stream request | query=%s | llm_engine=%s | user=%s",
        clean_query,
        llm_engine,
        getattr(user, "id", None) if user is not None else None,
    )

    if user and getattr(user, "custom_instructions", None):
        logger.info("USER GEMS LOADED: %s", user.custom_instructions)
    elif user:
        logger.info("USER AUTHENTICATED but no GEMS found.")
    else:
        logger.info("ANONYMOUS REQUEST (no GEMS).")

    return StreamingResponse(
        agent_event_generator(request, clean_query, llm_engine, user),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )