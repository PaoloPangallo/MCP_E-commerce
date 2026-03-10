from __future__ import annotations

from app.db.database import SessionLocal
from app.mcp.server import configure_mcp, mcp

configure_mcp(db_factory=SessionLocal)

app = mcp.streamable_http_app()