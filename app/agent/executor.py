from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, Iterable, List, Optional

from app.db.redis import redis_client
from app.config.cache import TOOL_EXECUTOR_TTL
from app.agent.schemas import Observation, ObservationQuality, ObservationStatus, ToolCall
from app.mcp.client import MCPToolClient

logger = logging.getLogger(__name__)


class ToolExecutor:
    _CACHE_TTL = TOOL_EXECUTOR_TTL

    def __init__(
        self,
        mcp_client: Optional[MCPToolClient] = None,
        context: Optional[Any] = None, # mantengo la firma per retrocompatibilità momentanea
        prefer_mcp: bool = True,
        fallback_to_local: bool = False,
        mcp_mode: str = "standard",
    ) -> None:
        self.context = context
        self.mcp_client = mcp_client
        self.mcp_mode = mcp_mode
        self._mcp_tools: Optional[set[str]] = None

    async def initialize(self) -> None:
        if self.mcp_client and getattr(self.mcp_client, "is_available", True):
            try:
                tools_list = await self.mcp_client.list_tools_async()
                self._mcp_tools = set(tools_list)
                logger.info("ToolExecutor discovered MCP tools: %s", self._mcp_tools)
            except Exception as exc:
                logger.warning("ToolExecutor failed to list MCP tools: %s", exc)
                self._mcp_tools = None

    async def execute(self, tool_call: ToolCall) -> Observation:
        if self._mcp_tools is None and self.mcp_client:
            await self.initialize()

        if self._mcp_tools is not None and tool_call.tool not in self._mcp_tools:
            logger.warning("Unknown MCP tool requested: %s", tool_call.tool)
            return Observation(
                tool=tool_call.tool,
                ok=False,
                status="error",
                error=f"Unknown Tool '{tool_call.tool}'",
                summary=f"Tool '{tool_call.tool}' non trovato nel catalogo MCP.",
                retryable=False,
                state_key=tool_call.tool,
                terminal=False,
                quality="empty",
            )

        cache_key = self._make_cache_key(tool_call.tool, tool_call.input, self.mcp_mode)
        
        # In modalità Playwright l'ambiente è stateful, le azioni non devono essere mai cachate
        # altrimenti se l'LLM ripete un'azione (es. browser_type) ottiene il finto "ok" senza interagire col browser
        can_cache = self.mcp_mode != "playwright_browser"
        
        if can_cache:
            cached = self._get_cached_result(cache_key)
            if cached is not None:
                logger.info("ToolExecutor cache hit | tool=%s", tool_call.tool)
                return self._build_observation(
                    tool_call=tool_call,
                    result=dict(cached),
                    execution_ms=0.0,
                    cache_hit=True,
                )

        start = time.perf_counter()
        try:
            logger.info("ToolExecutor using MCP | tool=%s", tool_call.tool)
            
            # Inject session_id se presente nel context (utile per logging o persistenza lato tool)
            if self.context and getattr(self.context, "user", None):
                user_id = getattr(self.context.user, "id", None)
                if user_id:
                    tool_call.input["session_id"] = str(user_id)

            if not self.mcp_client:
                raise RuntimeError("MCP client is not initialized")

            result = await self.mcp_client.call_tool_async(tool_call.tool, tool_call.input)
            result = self._normalize_result_payload(result)
            result.setdefault("_backend", "mcp")

            execution_ms = round((time.perf_counter() - start) * 1000.0, 2)
            # Non cachare risultati di errore - potrebbero essere transitori
            if can_cache and result.get("status") != "error":
                self._set_cached_result(cache_key, result)

            logger.info(
                "ToolExecutor MCP success | tool=%s | status=%s | execution_ms=%s",
                tool_call.tool,
                result.get("status"),
                execution_ms,
            )

            return self._build_observation(
                tool_call=tool_call,
                result=result,
                execution_ms=execution_ms,
                cache_hit=False,
            )

        except Exception as exc:
            logger.exception("ToolExecutor execution failed | tool=%s | error=%s", tool_call.tool, exc)
            return Observation(
                tool=tool_call.tool,
                ok=False,
                status="error",
                error=str(exc),
                summary=f"{tool_call.tool} failed: {exc}",
                retryable=False,
                state_key=tool_call.tool,
                terminal=False,
                quality="empty",
                data={"error": str(exc)},
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

    @classmethod
    def _get_cached_result(cls, cache_key: str) -> Optional[Dict[str, Any]]:
        return redis_client.get_json(cache_key)

    @classmethod
    def _set_cached_result(cls, cache_key: str, result: Dict[str, Any]) -> None:
        redis_client.set_json(cache_key, result, ttl_seconds=int(cls._CACHE_TTL))

    _CACHE_EXCLUDE_KEYS: frozenset = frozenset({"session_id", "conversation_history", "context_info"})

    @classmethod
    def _make_cache_key(cls, tool_name: str, action_input: Dict[str, Any], mcp_mode: str = "standard") -> str:
        semantic_input = {
            k: v for k, v in (action_input or {}).items()
            if k not in cls._CACHE_EXCLUDE_KEYS
        }
        return f"{mcp_mode}:{tool_name}:{json.dumps(semantic_input, sort_keys=True, ensure_ascii=False, default=str)}"

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
        result: Dict[str, Any],
        execution_ms: float,
        cache_hit: bool,
    ) -> Observation:
        
        raw_status = result.get("status", "ok")
        # Map raw statuses from tools (like 'open', 'closed') to allowed Pydantic literals
        status = raw_status if raw_status in {"ok", "no_data", "error"} else ("error" if raw_status == "error" else "ok")
        
        # "no_data" = tool ran OK but found nothing → empty quality so agent can consider retrying
        quality = "good" if status == "ok" else "empty"
        # il browser si è già aperto, non ha senso riprovare.
        # IMPORTANTE: In un flusso multi-step, lo status "ok" NON deve mai essere terminale!
        playwright_terminal_statuses = {
            "contact_button_not_found", "login_required",
            "message_form_not_found", "submit_button_not_found", "message_sent",
        }
        contact_status = result.get("contact_status", "")
        terminal = bool(result.get("terminal", False)) or (contact_status in playwright_terminal_statuses)
        summary = result.get("summary", f"Tool {tool_call.tool} completato con status {status}.")

        # For browser tools, enrich the summary with actual page content so the LLM
        # can "see" what is on the page (page_text is extracted from the DOM by BrowserManager).
        _browser_view_tools = {"browser_navigate", "browser_type", "browser_get_view", "browser_click"}
        if tool_call.tool in _browser_view_tools and status == "ok" and not result.get("summary"):
            title = result.get("title", "")
            url = result.get("url", "")
            page_text = result.get("page_text", "")
            if page_text:
                snippet = page_text[:700]
                summary = f"Browser: '{title}' | {url}\n---\n{snippet}"
            elif url:
                summary = f"Browser: '{title}' | {url} (nessun testo estratto)"

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
            state_key=tool_call.tool,
            state_update=result,
            terminal=terminal,
            quality=quality,
            execution_ms=execution_ms,
            cache_hit=cache_hit,
        )