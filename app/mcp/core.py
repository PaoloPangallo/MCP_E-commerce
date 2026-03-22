import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Annotated

from pydantic import Field
from mcp.server.fastmcp import FastMCP
from app.models.user import User

logger = logging.getLogger(__name__)

# ============================================================
# MCP CONFIG
# ============================================================

@dataclass
class MCPDependencies:
    db_factory: Optional[Callable[[], Any]] = None
    user_resolver: Optional[Callable[[Any], Optional[object]]] = None

@dataclass
class MCPToolContext:
    db: Any
    user: Optional[object] = None
    llm_engine: str = "gemini"

_DEPS = MCPDependencies()

# THE Central MCP Instance
mcp = FastMCP("mcp-ecommerce-agent")

def configure_mcp(
    db_factory: Optional[Callable[[], Any]] = None,
    user_resolver: Optional[Callable[[Any], Optional[object]]] = None,
) -> None:
    _DEPS.db_factory = db_factory
    _DEPS.user_resolver = user_resolver
    logger.info("MCP configured | db_factory=%s | user_resolver=%s", bool(db_factory), bool(user_resolver))

# ============================================================
# INTERNAL HELPERS
# ============================================================

def _get_db() -> Any:
    if _DEPS.db_factory is None:
        return None
    return _DEPS.db_factory()

def _close_db(db: Any) -> None:
    if db is None:
        return

    close = getattr(db, "close", None)
    if callable(close):
        try:
            close()
        except Exception as exc:
            logger.warning("Failed to close DB session: %s", exc)

from contextlib import contextmanager

@contextmanager
def _db_context():
    """Context manager garantisce la chiusura della sessione DB anche in caso di eccezione."""
    db = _get_db()
    try:
        yield db
    finally:
        _close_db(db)

def resolve_user_by_id(user_id_str: str) -> Optional[User]:
    """Default user resolver that fetches user from DB by ID string."""
    if not user_id_str:
        return None
    
    db = _get_db()
    if not db:
        return None
        
    try:
        user_id = int(user_id_str)
        user = db.query(User).filter(User.id == user_id).first()
        return user
    except (ValueError, Exception) as exc:
        logger.warning("Failed to resolve user by ID '%s': %s", user_id_str, exc)
        return None
    finally:
        _close_db(db)

def _build_context(db: Any, llm_engine: str = "gemini", session_id: Optional[str] = None) -> MCPToolContext:
    user = None
    
    # Try configured resolver first
    if session_id and _DEPS.user_resolver:
        try:
            user = _DEPS.user_resolver(session_id)
        except Exception as exc:
            logger.warning("Failed to resolve user for session_id=%s via custom resolver: %s", session_id, exc)

    # Fallback to default ID resolver if still None
    if user is None and session_id:
        user = resolve_user_by_id(session_id)

    return MCPToolContext(
        db=db,
        user=user,
        llm_engine=(llm_engine or "ollama").strip().lower(),
    )

def _tool_error(**kwargs: Any) -> Dict[str, Any]:
    payload = {"status": "error", **kwargs}
    return payload
