from __future__ import annotations

from dotenv import load_dotenv

from app.db.database import SessionLocal
from app.mcp.server import configure_mcp, mcp

load_dotenv()

configure_mcp(db_factory=SessionLocal)

# Con la tua versione del pacchetto MCP:
# - NON usare streamable_http_path="/"
# - il transport espone già /mcp come path interno di default
# Quindi, servendo questa app direttamente su :8050,
# l'endpoint finale sarà http://127.0.0.1:8050/mcp
app = mcp.streamable_http_app()