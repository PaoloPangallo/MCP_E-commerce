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

    async def list_tools_async(self) -> List[str]:
        self._ensure_ready()
        logger.info("MCP client list_tools_async | local=%s", self.is_local)
        
        if self.is_local:
            from app.mcp.server import mcp
            tools_list = await mcp.list_tools()
            return [t.name for t in tools_list]

        assert self._session is not None
        tools = await self._session.list_tools()
        return [tool.name for tool in tools.tools]

    async def get_tool_schemas_async(self) -> Dict[str, Dict[str, Any]]:
        self._ensure_ready()
        logger.info("MCP client get_tool_schemas_async | local=%s", self.is_local)
        
        catalog = {}
        if self.is_local:
            from app.mcp.server import mcp
            mcp_tools = await mcp.list_tools()
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
        print(f"\n>>> [MCP] CALL TOOL: {tool_name} | args={arguments}")
        logger.info("MCP client call_tool_async | local=%s | tool=%s", self.is_local, tool_name)

        try:
            if self.is_local:
                from app.mcp.server import mcp
                # Natively call underlying execution
                result_content = await mcp.call_tool(tool_name, arguments or {})
                print(f">>> [MCP] Local Result Content: {str(result_content)[:200]}...")
                
                # FastMCP in newer versions might return a list of Content objects
                # or a CallToolResult object.
                if hasattr(result_content, "content"):
                     content = result_content.content
                elif isinstance(result_content, list):
                     content = result_content
                elif isinstance(result_content, tuple):
                     # Se è una tuple, di solito il primo elemento è la lista di content
                     content = result_content[0] if len(result_content) > 0 else []
                else:
                     content = [result_content]
            else:
                result = await self._session.call_tool(tool_name, arguments or {})
                content = getattr(result, "content", None)

            if not content:
                logger.info("MCP call_tool_async empty content | tool=%s", tool_name)
                return {"status": "ok", "result": None, "_backend": "mcp"}

            parts: List[str] = []
            if isinstance(content, list):
                for item in content:
                    # 1. Proviamo come oggetto (TextContent)
                    text = getattr(item, "text", None)
                    if text is not None:
                        parts.append(text)
                    # 2. Proviamo come dict
                    elif isinstance(item, dict) and "text" in item:
                        parts.append(item["text"])
                    # 3. Se l'item stesso è una stringa (es. se FastMCP ha restituito list[str])
                    elif isinstance(item, str):
                        parts.append(item)
                    # 4. Fallback estremo: se l'item è un dict ma NON ha "text", 
                    # potrebbe essere il risultato RAW del tool che FastMCP non ha inscatolato.
                    elif isinstance(item, dict):
                        parts.append(json.dumps(item))
            elif isinstance(content, str):
                parts.append(content)
            elif isinstance(content, dict):
                parts.append(json.dumps(content))

            if not parts:
                return {"status": "ok", "result": None, "_backend": "mcp"}

            joined: str = "\n".join(parts).strip()
            print(f">>> [MCP] Final JOINED for parsing: {joined[:100]}...")
            
            try:
                parsed_result: Any = json.loads(joined)
                print(f"[MCP] JSON parsed successfully. Type: {type(parsed_result)}")
                if isinstance(parsed_result, dict):
                    parsed_result["_backend"] = "mcp"
                    return parsed_result
                return {"status": "ok", "result": parsed_result, "_backend": "mcp"}
            except Exception as e:
                print(f"[MCP] JSON parse FAILED: {e}")
                # Se il parsing fallisce, proviamo a vedere se è un dict-string python (con apici singoli)
                # Questo succede se FastMCP non serializza correttamente in JSON in modalità locale
                if joined.startswith("{") and joined.endswith("}"):
                    try:
                        import ast
                        evaluated = ast.literal_eval(joined)
                        if isinstance(evaluated, dict):
                             print("[MCP] ast.literal_eval SUCCEEDED for dict-string")
                             evaluated["_backend"] = "mcp"
                             return evaluated
                    except:
                        pass
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
                # FastMCP supports reading resources by URI
                try:
                    # In FastMCP, read_resource usually returns a list of ResourceContent
                    content = await mcp.read_resource(uri)
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
                from app.mcp.server import mcp
                try:
                    # FastMCP supports getting prompts by name
                    # Returns a GetPromptResult which has 'messages'
                    result = await mcp.get_prompt(name)
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