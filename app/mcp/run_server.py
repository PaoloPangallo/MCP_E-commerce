from __future__ import annotations

import logging
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting standalone MCP server on http://127.0.0.1:8050/mcp")
    uvicorn.run(
        "app.mcp.asgi:app",
        host="127.0.0.1",
        port=8050,
        reload=False,
    )