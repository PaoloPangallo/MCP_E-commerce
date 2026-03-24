from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel

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

WORKER_HARD_TIMEOUT_SECONDS = 380.0
QUEUE_WAIT_TIMEOUT_SECONDS = 75.0
HEARTBEAT_INTERVAL_SECONDS = 5.0

_ALLOWED_EVENT_TYPES = {
    "start",
    "thinking",
    "tool_start",
    "tool_result",
    "final",
    "error",
    "heartbeat",
    "done",
    "answer_chunk",
    "vision_analysis",
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
    if llm_engine in {"gemini", "ollama", "ollama_cloud", "rule_based"}:
        return llm_engine
    return "ollama_cloud"


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

    # Truncate if necessary (using a method that avoids triggering certain IDE slicing rules)
    if len(q) > 500:
        return q[0:500].rstrip()

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


class _SafeEncoder(json.JSONEncoder):
    """Converte tipi non-serializzabili standard (numpy floats, nan, inf)."""
    def default(self, obj):
        # numpy scalar types
        try:
            import numpy as np
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return None if (math.isnan(float(obj)) or math.isinf(float(obj))) else float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        # Python floats nan/inf
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return super().default(obj)


def _sse(data: Dict[str, Any], event: str | None = None) -> str:
    try:
        payload = json.dumps(data, ensure_ascii=False, cls=_SafeEncoder)
    except Exception as e:
        # Stampiamo l'errore esatto su stdout cosicché l'utente possa vederlo
        print(f"\n[ERROR] SSE SERIALIZATION FAILED: {str(e)}")
        print(f"[DEBUG] Failed keys: {list(data.keys())}")
        
        logger.error("SSE serialization failed: %s | keys=%s", e, list(data.keys()))
        # Fallback estremo: emetti l'evento senza il campo "data" ma con una notifica di errore
        safe = {k: v for k, v in data.items() if k != "data"}
        safe["serialization_error"] = str(e)
        payload = json.dumps(safe, ensure_ascii=False, cls=_SafeEncoder)

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
    image: Optional[str] = None,
):
    llm_engine = _normalize_llm_engine(llm_engine)
    query = _sanitize_query(query)

    if not query and not image:
        yield _sse(
            {
                "type": "error",
                "message": "Query non valida o vuota.",
            }
        )
        yield _sse({"type": "done"})
        return

    done_sent = False
    
    # Use a queue to decouple agent execution from SSE yielding.
    # This ensures heartbeats are sent even if the agent is blocked by a tool call.
    queue = asyncio.Queue()

    async def run_agent():
        db = None
        try:
            db = SessionLocal()
            agent = EbayReactAgent(db=db, user=user)
            
            logger.info(
                "Agent created for stream [VER: QUEUE_HBEAT_FIX] | user=%s",
                getattr(user, "id", None)
            )
            agent_request = AgentRequest(
                query=query,
                llm_engine=llm_engine,
                max_steps=6,
                return_trace=True,
                image=image,
            )

            async with _stream_semaphore:
                async for event in agent.run_stream(agent_request):
                    await queue.put(event)
                    if event.get("type") == "done":
                        break
        except Exception as e:
            logger.exception("Error in background agent task")
            await queue.put({"type": "error", "message": str(e)})
        finally:
            await queue.put(None)  # Sentinel for completion
            if db is not None:
                try:
                    db.close()
                except Exception as exc:
                    logger.warning("Failed to close agent DB session: %s", exc)

    # Start agent in background
    agent_task = asyncio.create_task(run_agent())
    start_time = time.monotonic()
    
    try:
        while True:
            # Check for hard timeout
            if time.monotonic() - start_time > WORKER_HARD_TIMEOUT_SECONDS:
                logger.warning("Agent stream hit hard timeout of %ss", WORKER_HARD_TIMEOUT_SECONDS)
                yield _sse({"type": "error", "message": "Timeout della richiesta."})
                break

            # Check for client disconnection
            if await request.is_disconnected():
                logger.info("Client disconnected from /agent/stream")
                break

            try:
                # Wait for next event or heartbeat timeout
                msg = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_INTERVAL_SECONDS)
                
                if msg is None: # Agent finished
                    break
                
                validated = _validate_event(msg)
                event_type = validated.get("type", "unknown")
                print(f"[SSE] Yielding event: {event_type}")
                logger.debug("Yielding SSE event | type=%s", event_type)
                yield _sse(validated)
                
                if event_type == "done":
                    done_sent = True

            except asyncio.TimeoutError:
                # No messages for HEARTBEAT_INTERVAL_SECONDS, send heartbeat
                yield _sse({"type": "heartbeat"})

        if not done_sent and not await request.is_disconnected():
            yield _sse({"type": "done"})

    except asyncio.CancelledError:
        logger.info("SSE stream cancelled by server/client")
    except Exception as exc:
        logger.exception("SSE generator error: %s", exc)
        if not await request.is_disconnected():
            yield _sse({"type": "error", "message": str(exc)})

    finally:
        # Ensure task is cleaned up
        if not agent_task.done():
            agent_task.cancel()
            try:
                await agent_task
            except asyncio.CancelledError:
                pass


class StreamRequest(BaseModel):
    query: str
    llm_engine: str = "ollama_cloud"
    image: Optional[str] = None

async def _handle_agent_stream(request: Request, clean_query: str, llm_engine: str, user: Any, image: Optional[str] = None):

    if _looks_like_event_stream_payload(clean_query):
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
        agent_event_generator(request, clean_query, llm_engine, user, image),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@router.post("/stream")
async def agent_stream_post(
    body: StreamRequest,
    request: Request,
    user=Depends(get_optional_user)
):
    clean_query = _sanitize_query(body.query)
    if not clean_query and not body.image:
        raise HTTPException(status_code=400, detail="Query o immagine necessaria.")
    
    return await _handle_agent_stream(request, clean_query, body.llm_engine, user, body.image)

@router.get("/stream")
async def agent_stream(
    request: Request,
    query: str = Query(..., min_length=1),
    llm_engine: str = Query("ollama_cloud"),
    user=Depends(get_optional_user),
):
    clean_query = _sanitize_query(query)

    if not clean_query:
        raise HTTPException(status_code=400, detail="Query non valida o vuota.")

    return await _handle_agent_stream(request, clean_query, llm_engine, user)
