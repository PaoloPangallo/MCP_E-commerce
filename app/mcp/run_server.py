from __future__ import annotations

import os

import uvicorn
from dotenv import load_dotenv

load_dotenv()


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    return value in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    uvicorn.run(
        "app.mcp.asgi:app",
        host=os.getenv("MCP_HOST", "127.0.0.1"),
        port=int(os.getenv("MCP_PORT", "8050")),
        reload=_env_flag("MCP_RELOAD", True),
        log_level=os.getenv("MCP_LOG_LEVEL", "info"),
    )