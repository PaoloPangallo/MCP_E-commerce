from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, Iterable, List, Optional

from app.agent.schemas import Observation, ObservationQuality, ObservationStatus, ToolCall
from app.agent.tool_registry import TOOLS, ToolContext, get_tool_spec
from app.mcp.client import MCPToolClient

logger = logging.getLogger(__name__)


class ToolExecutor:
    _RESULT_CACHE: Dict[str, tuple[float, Dict[str, Any]]] = {}
    _CACHE_TTL = 120.0

    def __init__(
        self,
        context: ToolContext,
        mcp_client: Optional[MCPToolClient] = None,
        prefer_mcp: bool = True,
        fallback_to_local: bool = True,
    ) -> None:
        self.context = context
        self.mcp_client = mcp_client
        self.prefer_mcp = bool(prefer_mcp)
        self.fallback_to_local = bool(fallback_to_local)

    async def execute(self, tool_call: ToolCall) -> Observation:
        spec = TOOLS.get(tool_call.tool)
        if spec is None:
            logger.warning("Unknown tool requested: %s", tool_call.tool)
            return Observation(
                tool=tool_call.tool,
                ok=False,
                status="error",
                error=f"Unknown tool '{tool_call.tool}'",
                summary=f"Tool '{tool_call.tool}' non disponibile.",
                retryable=False,
                state_key=None,
                terminal=False,
                quality="empty",
            )

        cache_key = self._make_cache_key(tool_call.tool, tool_call.input)
        if spec.use_cache:
            cached = self._get_cached_result(cache_key)
            if cached is not None:
                logger.info("ToolExecutor cache hit | tool=%s", tool_call.tool)
                return self._build_observation(
                    tool_call=tool_call,
                    spec=spec,
                    result=dict(cached),
                    execution_ms=0.0,
                    cache_hit=True,
                )

        attempts = max(1, int(spec.max_retries) + 1)
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            start = time.perf_counter()

            try:
                result = await self._execute_once(tool_call=tool_call, spec=spec)

                result = self._normalize_result_payload(result)
                result["_tool_attempts"] = attempt

                if spec.result_normalizer:
                    result = spec.result_normalizer(result, tool_call.input)
                    result = self._normalize_result_payload(result)
                    result.setdefault("_tool_attempts", attempt)

                execution_ms = round((time.perf_counter() - start) * 1000.0, 2)

                if spec.use_cache:
                    self._set_cached_result(cache_key, result)

                logger.info(
                    "ToolExecutor success | tool=%s | backend=%s | status=%s | execution_ms=%s",
                    tool_call.tool,
                    result.get("_backend"),
                    result.get("status"),
                    execution_ms,
                )

                return self._build_observation(
                    tool_call=tool_call,
                    spec=spec,
                    result=result,
                    execution_ms=execution_ms,
                    cache_hit=False,
                )

            except Exception as exc:
                last_error = exc
                logger.exception(
                    "ToolExecutor attempt failed | tool=%s | attempt=%s/%s | error=%s",
                    tool_call.tool,
                    attempt,
                    attempts,
                    exc,
                )
                if attempt >= attempts:
                    break

        return Observation(
            tool=tool_call.tool,
            ok=False,
            status="error",
            error=str(last_error) if last_error else "Unknown tool execution error",
            summary=f"{tool_call.tool} failed: {last_error}" if last_error else f"{tool_call.tool} failed",
            retryable=False,
            state_key=spec.state_key or None,
            terminal=False,
            quality="empty",
            data={
                "_backend": None,
                "error": str(last_error) if last_error else "Unknown tool execution error",
            },
            state_update={
                "_backend": None,
                "error": str(last_error) if last_error else "Unknown tool execution error",
            },
            execution_ms=None,
            cache_hit=False,
        )

    async def execute_many(self, tool_calls: Iterable[ToolCall], parallel: bool = False) -> List[Observation]:
        calls = list(tool_calls)
        if not calls:
            return []

        if not parallel or len(calls) <= 1:
            results = []
            for call in calls:
                results.append(await self.execute(call))
            return results

        logger.info("ToolExecutor execute_many | parallel=%s | count=%s", parallel, len(calls))

        tasks = [self.execute(call) for call in calls]
        return list(await asyncio.gather(*tasks, return_exceptions=False))

    async def _execute_once(self, tool_call: ToolCall, spec: Any) -> Dict[str, Any]:
        if self._should_use_mcp(tool_call.tool):
            try:
                logger.info(
                    "ToolExecutor using MCP | tool=%s | input=%s",
                    tool_call.tool,
                    tool_call.input,
                )
                assert self.mcp_client is not None
                result = await self.mcp_client.call_tool_async(tool_call.tool, tool_call.input)
                result = self._normalize_result_payload(result)
                result.setdefault("_backend", "mcp")
                logger.info("ToolExecutor MCP success | tool=%s", tool_call.tool)
                return result

            except Exception as exc:
                logger.exception(
                    "ToolExecutor MCP failed | tool=%s | error=%s",
                    tool_call.tool,
                    exc,
                )
                if not self.fallback_to_local:
                    raise

                logger.warning("ToolExecutor fallback to local | tool=%s", tool_call.tool)

        logger.info(
            "ToolExecutor using LOCAL backend | tool=%s | input=%s",
            tool_call.tool,
            tool_call.input,
        )
        result = await asyncio.to_thread(spec.executor, tool_call.input, self.context)
        result = self._normalize_result_payload(result)
        result.setdefault("_backend", "local")
        logger.info("ToolExecutor LOCAL success | tool=%s", tool_call.tool)
        return result

    def _should_use_mcp(self, tool_name: str) -> bool:
        if not self.prefer_mcp:
            return False

        if self.mcp_client is None:
            return False

        is_available = getattr(self.mcp_client, "is_available", True)
        if not is_available:
            return False

        return tool_name in {
            "search_products",
            "analyze_seller",
            "conversation",
            "profile_query",
            "compare_products",
            "get_item_details",
            "get_shipping_costs",
        }

    @classmethod
    def _get_cached_result(cls, cache_key: str) -> Optional[Dict[str, Any]]:
        cached_entry = cls._RESULT_CACHE.get(cache_key)
        if not cached_entry:
            return None

        ts, cached = cached_entry
        if time.time() - ts >= cls._CACHE_TTL:
            try:
                del cls._RESULT_CACHE[cache_key]
            except KeyError:
                pass
            return None

        return dict(cached)

    @classmethod
    def _set_cached_result(cls, cache_key: str, result: Dict[str, Any]) -> None:
        cls._RESULT_CACHE[cache_key] = (time.time(), dict(result))

    @staticmethod
    def _make_cache_key(tool_name: str, action_input: Dict[str, Any]) -> str:
        return f"{tool_name}:{json.dumps(action_input or {}, sort_keys=True, ensure_ascii=False, default=str)}"

    @staticmethod
    def _normalize_result_payload(result: Any) -> Dict[str, Any]:
        if isinstance(result, dict):
            return result

        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict):
                    return parsed
                return {"result": parsed}
            except Exception:
                return {"result": result}

        return {"result": result}

    @staticmethod
    def _build_observation(
        tool_call: ToolCall,
        spec: Any,
        result: Dict[str, Any],
        execution_ms: float,
        cache_hit: bool,
    ) -> Observation:
        status: ObservationStatus = "ok"
        if spec.status_resolver:
            status = spec.status_resolver(result)

        quality: ObservationQuality = "good"
        if spec.quality_resolver:
            quality = spec.quality_resolver(result)

        terminal = False
        if spec.terminal_resolver:
            terminal = spec.terminal_resolver(result)

        summary = spec.summarizer(result) if spec.summarizer else "Tool eseguito."
        if cache_hit:
            summary = f"[cache] {summary}"

        return Observation(
            tool=tool_call.tool,
            ok=status != "error",
            status=status,
            data=result,
            summary=summary,
            error=result.get("error"),
            retryable=False,
            state_key=spec.state_key or None,
            state_update=result,
            terminal=terminal,
            quality=quality,
            execution_ms=execution_ms,
            cache_hit=cache_hit,
        )