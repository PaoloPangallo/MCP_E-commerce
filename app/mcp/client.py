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
        self.is_local = "127.0.0.1" in self.server_url or "localhost" in self.server_url
        
        self._exit_stack = None
        self._session = None

    @property
    def is_available(self) -> bool:
        return self.enabled

    async def __aenter__(self):
        if not self.enabled:
            return self

        if self.is_local:
            logger.info("MCP client using Local Direct Mode | server=%s", self.server_url)
            return self

        if _MCP_IMPORT_ERROR is not None:
            logger.warning("MCP client not available due to import error: %s", _MCP_IMPORT_ERROR)
            return self

        try:
            from contextlib import AsyncExitStack
            self._exit_stack = AsyncExitStack()
            
            read_stream, write_stream, _ = await self._exit_stack.enter_async_context(
                streamable_http_client(self.server_url)
            )
            
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            
            await self._session.initialize()
            logger.info("MCP client connected via HTTP | server=%s", self.server_url)
            
        except Exception as exc:
            logger.error("Failed to connect to MCP server via HTTP at %s: %s", self.server_url, exc)
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
        if self.is_local:
            return
        if _MCP_IMPORT_ERROR is not None:
            raise RuntimeError(f"MCP client non disponibile: {_MCP_IMPORT_ERROR}")
        if self._session is None:
             raise RuntimeError("MCP ClientSession is not initialized.")

    async def list_tools_async(self) -> List[str]:
        self._ensure_ready()
        logger.info("MCP client list_tools_async | local=%s", self.is_local)
        
        if self.is_local:
            from app.mcp.server import mcp
            return list(mcp._tools.keys())

        tools = await self._session.list_tools()
        return [tool.name for tool in tools.tools]

    async def get_tool_schemas_async(self) -> Dict[str, Dict[str, Any]]:
        self._ensure_ready()
        logger.info("MCP client get_tool_schemas_async | local=%s", self.is_local)
        
        catalog = {}
        if self.is_local:
            from app.mcp.server import mcp
            for name, tool in mcp._tools.items():
                # FastMCP internal parameter extraction
                parameters = getattr(tool, "parameters", {})
                if not parameters and hasattr(tool, "schema"):
                    parameters = tool.schema
                catalog[name] = {
                    "name": name,
                    "description": tool.description,
                    "input_schema": parameters or {}
                }
            return catalog

        tools = await self._session.list_tools()
        for tool in tools.tools:
            catalog[tool.name] = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": getattr(tool, "inputSchema", {}) or {},
            }
        return catalog

    async def call_tool_async(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._ensure_ready()
        logger.info("MCP client call_tool_async | local=%s | tool=%s", self.is_local, tool_name)

        try:
            if self.is_local:
                from app.mcp.server import mcp
                # Natively call underlying execution
                result_content = await mcp.call_tool(tool_name, arguments or {})
                # FastMCP call_tool returns a list of contents, likely TextContent
                content = result_content
            else:
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

    async def read_resource_async(self, uri: str) -> Optional[str]:
        self._ensure_ready()
        logger.info("MCP client read_resource_async | local=%s | uri=%s", self.is_local, uri)
        try:
            if self.is_local:
                from app.mcp.server import mcp
                # Not fully supported seamlessly in FastMCP wrapper bypassing, so we fallback gracefully
                return None
                
            result = await self._session.read_resource(uri)
            content = getattr(result, "contents", None)
            if not content:
                return None
            
            parts = []
            for item in content:
                text = getattr(item, "text", None)
                if text is not None:
                    parts.append(text)
            return "\n".join(parts).strip() if parts else None
        except Exception as exc:
            logger.error("MCP read_resource_async failed | uri=%s | error=%s", uri, exc)
            return None

    async def get_prompt_async(self, name: str) -> Optional[str]:
        self._ensure_ready()
        logger.info("MCP client get_prompt_async | local=%s | prompt=%s", self.is_local, name)
        try:
            if self.is_local:
                from app.mcp.server import mcp
                return None

            result = await self._session.get_prompt(name)
            messages = getattr(result, "messages", None)
            if not messages:
                return None
            
            parts = []
            for msg in messages:
                content = getattr(msg, "content", None)
                if getattr(content, "type", "") == "text":
                    parts.append(getattr(content, "text", ""))
            return "\n".join(parts).strip() if parts else None
        except Exception as exc:
            logger.error("MCP get_prompt_async failed | prompt=%s | error=%s", name, exc)
            return None