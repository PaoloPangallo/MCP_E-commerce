from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================
# MCP RESULT PARSING HELPERS
# ============================================================

def _extract_content_parts(raw_content: Any) -> List[str]:
    """
    Normalize any FastMCP content object into a flat list of text strings.
    Handles: CallToolResult, list[TextContent], list[dict], list[str], str, dict.
    """
    # Unwrap CallToolResult or any wrapper with .content
    if hasattr(raw_content, "content"):
        raw_content = raw_content.content
    # Unwrap tuple — FastMCP sometimes wraps (content_list, metadata)
    if isinstance(raw_content, tuple):
        raw_content = raw_content[0] if raw_content else []

    parts: List[str] = []
    if isinstance(raw_content, list):
        for item in raw_content:
            text = getattr(item, "text", None)
            if text is not None:
                parts.append(text)
            elif isinstance(item, dict):
                parts.append(item.get("text") or json.dumps(item))
            elif isinstance(item, str):
                parts.append(item)
    elif isinstance(raw_content, str):
        parts.append(raw_content)
    elif isinstance(raw_content, dict):
        parts.append(json.dumps(raw_content))
    return parts


def _parse_parts_to_dict(parts: List[str]) -> Dict[str, Any]:
    """
    Join content parts and deserialize to a dict.
    Tries JSON first, then ast.literal_eval for Python-style dicts.
    Falls back to wrapping the raw string.
    """
    if not parts:
        return {"status": "ok", "result": None, "_backend": "mcp"}

    joined = "\n".join(parts).strip()
    logger.debug("MCP parsing joined content (%d chars): %s...", len(joined), joined[:120])

    # 1. JSON parse (preferred)
    try:
        parsed: Any = json.loads(joined)
        if isinstance(parsed, dict):
            parsed["_backend"] = "mcp"
            return parsed
        return {"status": "ok", "result": parsed, "_backend": "mcp"}
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Python dict string (single quotes) — FastMCP local mode quirk
    if joined.startswith("{") and joined.endswith("}"):
        try:
            evaluated = ast.literal_eval(joined)
            if isinstance(evaluated, dict):
                logger.debug("MCP ast.literal_eval succeeded")
                evaluated["_backend"] = "mcp"
                return evaluated
        except (ValueError, SyntaxError):
            pass

    # 3. Raw string fallback
    return {"status": "ok", "result": joined, "_backend": "mcp"}

try:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    _MCP_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:
    # Use class mocks to satisfy Pyre's type form requirements
    class ClientSession: pass
    class streamable_http_client: pass
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
        
        from contextlib import AsyncExitStack
        self._exit_stack: Optional[AsyncExitStack] = None
        self._session: Optional[ClientSession] = None

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
            stack = AsyncExitStack()
            self._exit_stack = stack
            
            # Use local variables to avoid Optional[None] Pyre errors
            if streamable_http_client is None:
                raise RuntimeError("mcp streamable_http_client not available")

            read_stream, write_stream, _ = await stack.enter_async_context(
                streamable_http_client(self.server_url)
            )
            
            session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            self._session = session
            
            if session:
                await session.initialize()
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

    def _get_local_mcp_instance(self) -> Any:
        """Resolve the FastMCP instance based on the server_url path."""
        url = self.server_url.lower()
        if "/playwright" in url:
            from app.mcp.playwright_server import mcp_playwright
            return mcp_playwright
        else:
            from app.mcp.server import mcp
            return mcp

    async def list_tools_async(self) -> List[str]:
        self._ensure_ready()
        logger.info("MCP client list_tools_async | local=%s", self.is_local)
        
        if self.is_local:
            mcp_instance = self._get_local_mcp_instance()
            tools_list = await mcp_instance.list_tools()
            return [t.name for t in tools_list]

        assert self._session is not None
        tools = await self._session.list_tools()
        return [tool.name for tool in tools.tools]

    async def get_tool_schemas_async(self) -> Dict[str, Dict[str, Any]]:
        self._ensure_ready()
        logger.info("MCP client get_tool_schemas_async | local=%s", self.is_local)
        
        catalog = {}
        if self.is_local:
            mcp_instance = self._get_local_mcp_instance()
            mcp_tools = await mcp_instance.list_tools()
            for tool in mcp_tools:
                # Standard MCP Tool object attributes
                parameters = getattr(tool, "inputSchema", {})
                if hasattr(parameters, "model_dump"):
                    parameters = parameters.model_dump()
                
                catalog[tool.name] = {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": parameters or {}
                }
            return catalog

        assert self._session is not None
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
        logger.debug("MCP CALL TOOL: %s | args=%s", tool_name, arguments)
        logger.info("MCP client call_tool_async | local=%s | tool=%s", self.is_local, tool_name)

        try:
            if self.is_local:
                mcp_instance = self._get_local_mcp_instance()
                raw = await mcp_instance.call_tool(tool_name, arguments or {})
                logger.debug("MCP local raw result: %s...", str(raw)[:200])
            else:
                raw = await self._session.call_tool(tool_name, arguments or {})

            parts = _extract_content_parts(raw)
            if not parts:
                logger.info("MCP call_tool_async empty content | tool=%s", tool_name)
                return {"status": "ok", "result": None, "_backend": "mcp"}

            return _parse_parts_to_dict(parts)

        except Exception as exc:
            logger.error("MCP call_tool_async failed | tool=%s | error=%s", tool_name, exc)
            return {
                "status": "error",
                "error": str(exc),
                "_backend": "mcp",
            }

    async def read_resource_async(self, uri: str) -> Optional[str]:
        self._ensure_ready()
        logger.info("MCP client read_resource_async | local=%s | uri=%s", self.is_local, uri)
        try:
            if self.is_local:
                mcp_instance = self._get_local_mcp_instance()
                # FastMCP supports reading resources by URI
                try:
                    # In FastMCP, read_resource usually returns a list of ResourceContent
                    content = await mcp_instance.read_resource(uri)
                    if not content:
                        return None
                    parts = [getattr(item, "text", "") for item in content if hasattr(item, "text")]
                    return "\n".join(parts).strip() if parts else None
                except Exception as e:
                    logger.warning("Local MCP read_resource failed for %s: %s", uri, e)
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
                mcp_instance = self._get_local_mcp_instance()
                try:
                    # FastMCP supports getting prompts by name
                    # Returns a GetPromptResult which has 'messages'
                    result = await mcp_instance.get_prompt(name)
                    messages = getattr(result, "messages", None)
                    if not messages:
                        return None
                    parts = []
                    for msg in messages:
                        content = getattr(msg, "content", None)
                        if hasattr(content, "text"):
                            parts.append(content.text)
                    return "\n".join(parts).strip() if parts else None
                except Exception as e:
                    logger.warning("Local MCP get_prompt failed for %s: %s", name, e)
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