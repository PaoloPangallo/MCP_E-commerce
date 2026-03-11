from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    _MCP_IMPORT_ERROR: Exception | None = None
except Exception as exc:
    ClientSession = None
    streamable_http_client = None
    _MCP_IMPORT_ERROR = exc


class MCPToolClient:
    def __init__(
        self,
        server_url: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        self.server_url = server_url or os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8050/mcp/mcp")
        self.enabled = bool(enabled)
        
        self._exit_stack = None
        self._session = None

    @property
    def is_available(self) -> bool:
        return self.enabled and _MCP_IMPORT_ERROR is None

    async def __aenter__(self):
        if not self.enabled:
            return self

        if _MCP_IMPORT_ERROR is not None:
            logger.warning("MCP client not available due to import error: %s", _MCP_IMPORT_ERROR)
            return self

        try:
            from contextlib import AsyncExitStack
            self._exit_stack = AsyncExitStack()
            
            # Using streamable_http_client
            read_stream, write_stream, _ = await self._exit_stack.enter_async_context(
                streamable_http_client(self.server_url)
            )
            
            # Creating and initializing ClientSession
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            
            await self._session.initialize()
            logger.info("MCP client connected and initialized | server=%s", self.server_url)
            
        except Exception as exc:
            logger.error("Failed to connect to MCP server at %s: %s", self.server_url, exc)
            self._session = None
            if self._exit_stack:
                await self._exit_stack.aclose()
                self._exit_stack = None
            
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._session = None
            logger.info("MCP client connection closed | server=%s", self.server_url)

    def _ensure_ready(self) -> None:
        if not self.enabled:
            raise RuntimeError("MCP client disabled.")

        if _MCP_IMPORT_ERROR is not None:
            raise RuntimeError(
                f"MCP client non disponibile: {_MCP_IMPORT_ERROR}"
            )
            
        if self._session is None:
             raise RuntimeError("MCP ClientSession is not initialized. Make sure to use MCPToolClient as an async context manager.")

    async def list_tools_async(self) -> List[str]:
        self._ensure_ready()
        logger.info("MCP client list_tools_async | server=%s", self.server_url)
        tools = await self._session.list_tools()
        return [tool.name for tool in tools.tools]

    async def call_tool_async(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._ensure_ready()

        logger.info(
            "MCP client call_tool_async | server=%s | tool=%s | args=%s",
            self.server_url,
            tool_name,
            arguments or {},
        )

        try:
            result = await self._session.call_tool(tool_name, arguments or {})

            content = getattr(result, "content", None)
            if not content:
                logger.info("MCP call_tool_async empty content | tool=%s", tool_name)
                return {"status": "ok", "result": None, "_backend": "mcp"}

            parts: List[str] = []
            for item in content:
                text = getattr(item, "text", None)
                if text is not None:
                    parts.append(text)

            if not parts:
                return {"status": "ok", "result": None, "_backend": "mcp"}

            joined = "\n".join(parts).strip()
            
            try:
                parsed_result = json.loads(joined)
                if isinstance(parsed_result, dict):
                    parsed_result["_backend"] = "mcp"
                    return parsed_result
                return {"status": "ok", "result": parsed_result, "_backend": "mcp"}
            except Exception:
                return {"status": "ok", "result": joined, "_backend": "mcp"}
                
        except Exception as exc:
             logger.error("MCP call_tool_async failed | tool=%s | error=%s", tool_name, exc)
             return {
                 "status": "error",
                 "error": str(exc),
                 "_backend": "mcp"
             }