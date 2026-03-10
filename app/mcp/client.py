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
        self.server_url = server_url or os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp/mcp")
        self.enabled = bool(enabled)

    @property
    def is_available(self) -> bool:
        return self.enabled and _MCP_IMPORT_ERROR is None

    def list_tools(self) -> List[str]:
        self._ensure_ready()
        logger.info("MCP client list_tools | server=%s", self.server_url)
        return self._run_sync(self._list_tools_async())

    def call_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._ensure_ready()

        logger.info(
            "MCP client call_tool | server=%s | tool=%s | args=%s",
            self.server_url,
            tool_name,
            arguments or {},
        )

        result = self._run_sync(
            self._call_tool_async(
                tool_name=tool_name,
                arguments=arguments or {},
            )
        )

        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict):
                    result = parsed
                else:
                    result = {"result": parsed}
            except Exception:
                result = {"result": result}

        if not isinstance(result, dict):
            result = {"result": result}

        result.setdefault("_backend", "mcp")

        logger.info("MCP client call_tool success | tool=%s", tool_name)
        return result

    def _ensure_ready(self) -> None:
        if not self.enabled:
            raise RuntimeError("MCP client disabled.")

        if _MCP_IMPORT_ERROR is not None:
            raise RuntimeError(
                f"MCP client non disponibile: {_MCP_IMPORT_ERROR}"
            )

    import asyncio

    def _run_sync(self, coro):

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()

    async def _list_tools_async(self) -> List[str]:
        assert streamable_http_client is not None
        assert ClientSession is not None

        async with streamable_http_client(self.server_url) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools = await session.list_tools()
                return [tool.name for tool in tools.tools]

    async def _call_tool_async(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        assert streamable_http_client is not None
        assert ClientSession is not None

        async with streamable_http_client(self.server_url) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)

                content = getattr(result, "content", None)
                if not content:
                    return {"status": "ok", "result": None}

                parts: List[str] = []

                for item in content:
                    text = getattr(item, "text", None)
                    if text is not None:
                        parts.append(text)

                if not parts:
                    return {"status": "ok", "result": None}

                joined = "\n".join(parts).strip()

                try:
                    parsed = json.loads(joined)
                    if isinstance(parsed, dict):
                        return parsed
                    return {"result": parsed}
                except Exception:
                    return {"result": joined}