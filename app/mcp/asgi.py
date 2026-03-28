from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.db.database import SessionLocal
from app.mcp.server import configure_mcp, mcp
from app.mcp.playwright_server import configure_playwright_mcp, mcp_playwright

# Import tool registration for playwright world (side effect: registers contact_seller_playwright)
import app.mcp.tools.playwright_contact  # noqa: F401

# Configure both worlds with DB access
configure_mcp(db_factory=SessionLocal)
configure_playwright_mcp(db_factory=SessionLocal)

# Create ASGI sub-apps once — this also initialises session_manager on each instance
_standard_asgi = mcp.streamable_http_app()
_playwright_asgi = mcp_playwright.streamable_http_app()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start the task-group for each MCP session manager before accepting requests."""
    async with mcp.session_manager.run():
        async with mcp_playwright.session_manager.run():
            yield


# Root FastAPI app — both MCP worlds mounted as sub-apps
app = FastAPI(title="MCP Worlds Router", lifespan=lifespan)

app.mount("/standard", _standard_asgi)
app.mount("/playwright", _playwright_asgi)
