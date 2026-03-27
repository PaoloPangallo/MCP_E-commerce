from __future__ import annotations

from fastapi import FastAPI

from app.db.database import SessionLocal
from app.mcp.server import configure_mcp, mcp
from app.mcp.playwright_server import configure_playwright_mcp, mcp_playwright

# Import tool registration for playwright world (side effect: registers contact_seller_playwright)
import app.mcp.tools.playwright_contact  # noqa: F401

# Configure both worlds with DB access
configure_mcp(db_factory=SessionLocal)
configure_playwright_mcp(db_factory=SessionLocal)

# Root FastAPI app — both MCP worlds mounted as sub-apps
app = FastAPI(title="MCP Worlds Router")

app.mount("/standard", mcp.streamable_http_app())
app.mount("/playwright", mcp_playwright.streamable_http_app())
