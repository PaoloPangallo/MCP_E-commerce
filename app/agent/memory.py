from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.agent.schemas import Observation


def _compact_result(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": item.get("title"),
        "price": item.get("price"),
        "currency": item.get("currency"),
        "seller_name": item.get("seller_name") or item.get("seller_username"),
        "trust_score": item.get("trust_score"),
        "ranking_score": item.get("ranking_score") or item.get("_rerank_score"),
        "url": item.get("url"),
    }


def _safe_user_key(user: Optional[object]) -> str:
    if user is None:
        return "anonymous"

    for attr in ("id", "user_id", "username", "email"):
        value = getattr(user, attr, None)
        if value is not None and str(value).strip():
            return f"user:{value}"

    return f"obj:{id(user)}"


# ============================================================
# Query sanitation
# ============================================================

_SSE_PREFIX_RE = re.compile(r"^\s*data\s*:\s*\{", re.IGNORECASE)
_EVENT_STREAM_MARKERS = (
    '"type": "start"',
    '"type":"start"',
    '"type": "thinking"',
    '"type":"thinking"',
    '"type": "tool_start"',
    '"type":"tool_start"',
    '"type": "tool_result"',
    '"type":"tool_result"',
    '"type": "final"',
    '"type":"final"',
    '"type": "done"',
    '"type":"done"',
)
_JSONISH_RE = re.compile(r'^\s*[\{\[]')
_MULTILINE_RE = re.compile(r'\n{2,}')


def sanitize_user_query_for_memory(query: str) -> str:
    """
    Keep only genuine user text in memory.

    Reject or normalize:
    - SSE dumps like `data: {"type": ...}`
    - raw JSON payloads / event streams
    - huge multiline technical blobs
    """
    q = str(query or "").strip()
    if not q:
        return ""

    # obvious SSE/event-stream dump
    if _SSE_PREFIX_RE.search(q):
        return ""

    # multi-event stream accidentally pasted into memory
    lowered = q.lower()
    if "data:" in lowered and any(marker in lowered for marker in _EVENT_STREAM_MARKERS):
        return ""

    # raw json-ish technical payload
    if _JSONISH_RE.match(q) and any(marker in lowered for marker in _EVENT_STREAM_MARKERS):
        return ""

    # giant multiline technical blob
    if len(q) > 800 and (_MULTILINE_RE.search(q) or q.count("{") >= 3):
        return ""

    # collapse whitespace
    q = re.sub(r"\s+", " ", q).strip()

    # hard cap for memory cleanliness
    if len(q) > 300:
        q = q[:300].rstrip()

    return q


def _token_candidates_for_brand_hints(query: str) -> List[str]:
    q = sanitize_user_query_for_memory(query)
    if not q:
        return []

    tokens = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9_-]*", q)
    cleaned: List[str] = []

    for token in tokens:
        t = token.strip()
        if not t:
            continue

        lower = t.lower()

        # Skip very generic conversational / technical junk
        if lower in {
            "data", "type", "start", "thinking", "tool", "tool_start", "tool_result",
            "final", "done", "message", "query", "llm_engine", "max_steps",
            "planned_tasks", "agent_trace", "final_data", "session_memory",
            "long_term_memory", "recent_queries", "recent_sellers", "recent_products",
            "recent_tool_results", "errors", "metrics", "pending_tasks",
        }:
            continue

        # Skip tiny filler tokens
        if len(lower) < 3:
            continue

        cleaned.append(t)

    return cleaned


@dataclass
class SessionMemory:
    user_key: str
    recent_queries: List[str] = field(default_factory=list)
    recent_sellers: List[str] = field(default_factory=list)
    recent_products: List[Dict[str, Any]] = field(default_factory=list)
    recent_tool_results: List[Dict[str, Any]] = field(default_factory=list)

    def add_query(self, query: str, limit: int = 10) -> None:
        query = sanitize_user_query_for_memory(query)
        if not query:
            return

        self.recent_queries = [q for q in self.recent_queries if q != query]
        self.recent_queries.insert(0, query)
        self.recent_queries = self.recent_queries[:limit]

    def add_seller(self, seller_name: Optional[str], limit: int = 10) -> None:
        seller = str(seller_name or "").strip()
        if not seller:
            return
        self.recent_sellers = [s for s in self.recent_sellers if s.lower() != seller.lower()]
        self.recent_sellers.insert(0, seller)
        self.recent_sellers = self.recent_sellers[:limit]

    def add_products(self, products: List[Dict[str, Any]], limit: int = 10) -> None:
        cleaned = [_compact_result(item) for item in products[:limit] if isinstance(item, dict)]
        if not cleaned:
            return
        merged = cleaned + [item for item in self.recent_products if item not in cleaned]
        self.recent_products = merged[:limit]

    def add_tool_result(self, tool: str, summary: str, limit: int = 12) -> None:
        summary = str(summary or "").strip()
        if not summary:
            return
        self.recent_tool_results.insert(0, {"tool": tool, "summary": summary})
        self.recent_tool_results = self.recent_tool_results[:limit]

    def snapshot(self) -> Dict[str, Any]:
        return {
            "recent_queries": list(self.recent_queries[:5]),
            "recent_sellers": list(self.recent_sellers[:5]),
            "recent_products": list(self.recent_products[:5]),
            "recent_tool_results": list(self.recent_tool_results[:5]),
        }


@dataclass
class LongTermMemory:
    user_key: str
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    previous_searches: List[str] = field(default_factory=list)
    user_behaviour: Dict[str, Any] = field(default_factory=dict)

    def remember_search(self, query: str, limit: int = 30) -> None:
        query = sanitize_user_query_for_memory(query)
        if not query:
            return

        self.previous_searches = [q for q in self.previous_searches if q != query]
        self.previous_searches.insert(0, query)
        self.previous_searches = self.previous_searches[:limit]
        self.user_behaviour["search_count"] = int(self.user_behaviour.get("search_count", 0)) + 1

    def remember_seller(self, seller_name: Optional[str]) -> None:
        seller = str(seller_name or "").strip()
        if not seller:
            return
        sellers = list(self.user_preferences.get("favorite_sellers") or [])
        sellers = [s for s in sellers if str(s).lower() != seller.lower()]
        sellers.insert(0, seller)
        self.user_preferences["favorite_sellers"] = sellers[:10]

    def remember_brand_hint(self, query: str) -> None:
        tokens = _token_candidates_for_brand_hints(query)
        if not tokens:
            return

        brands = list(self.user_preferences.get("recent_brand_hints") or [])
        for token in tokens[:2]:
            brands = [b for b in brands if str(b).lower() != token.lower()]
            brands.insert(0, token)

        self.user_preferences["recent_brand_hints"] = brands[:10]

    def snapshot(self) -> Dict[str, Any]:
        return {
            "user_preferences": deepcopy(self.user_preferences),
            "previous_searches": list(self.previous_searches[:5]),
            "user_behaviour": deepcopy(self.user_behaviour),
        }


class MemoryService:
    """
    Small persistence facade.
    It is intentionally backend-agnostic: today it keeps state in memory,
    but its interface is stable enough to be swapped with Redis/DB/vector storage.
    """

    _session_store: Dict[str, SessionMemory] = {}
    _long_term_store: Dict[str, LongTermMemory] = {}

    def load_session_memory(self, user: Optional[object]) -> SessionMemory:
        user_key = _safe_user_key(user)
        stored = self._session_store.get(user_key)
        if stored is None:
            stored = SessionMemory(user_key=user_key)
            self._session_store[user_key] = stored
        return stored

    def load_long_term_memory(self, user: Optional[object]) -> LongTermMemory:
        user_key = _safe_user_key(user)
        stored = self._long_term_store.get(user_key)
        if stored is None:
            stored = LongTermMemory(user_key=user_key)
            self._long_term_store[user_key] = stored
        return stored

    def hydrate_request_state(self, user_query: str, user: Optional[object]) -> "RequestState":
        session_memory = self.load_session_memory(user)
        long_term_memory = self.load_long_term_memory(user)

        clean_query = sanitize_user_query_for_memory(user_query)

        # salviamo in memoria solo testo utente pulito
        if clean_query:
            session_memory.add_query(clean_query)
            long_term_memory.remember_search(clean_query)
            long_term_memory.remember_brand_hint(clean_query)

        return RequestState(
            user_query=str(user_query or "").strip(),
            session_memory=session_memory,
            long_term_memory=long_term_memory,
        )

    def persist_request_outcome(self, state: "RequestState", final_answer: str) -> None:
        state.final_answer = str(final_answer or "").strip()

        if state.last_seller_name:
            state.session_memory.add_seller(state.last_seller_name)
            state.long_term_memory.remember_seller(state.last_seller_name)

        if state.search_payload:
            products = state.search_payload.get("results") or []
            if isinstance(products, list):
                state.session_memory.add_products(products)

        for observation in state.observations[-5:]:
            state.session_memory.add_tool_result(observation.tool, observation.summary)


@dataclass
class RequestState:
    user_query: str
    session_memory: SessionMemory
    long_term_memory: LongTermMemory
    observations: List[Observation] = field(default_factory=list)
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    task_pointer: int = 0
    detected_intent: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    tool_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tool_call_counts: Dict[str, int] = field(default_factory=dict)
    llm_call_counts: Dict[str, int] = field(default_factory=dict)
    top_result: Optional[Dict[str, Any]] = None
    last_seller_name: Optional[str] = None
    search_analysis: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    final_answer: Optional[str] = None

    def load_tasks(self, tasks: List[Dict[str, Any]]) -> None:
        cleaned: List[Dict[str, Any]] = []

        for task in tasks or []:
            if not isinstance(task, dict):
                continue

            tool = str(task.get("tool") or "").strip()
            payload = task.get("input") or {}

            if not tool or not isinstance(payload, dict):
                continue

            normalized = {"tool": tool, "input": dict(payload)}
            if cleaned and cleaned[-1] == normalized:
                continue
            cleaned.append(normalized)

        self.tasks = cleaned
        self.task_pointer = 0

    def has_pending_tasks(self) -> bool:
        return self.task_pointer < len(self.tasks)

    def peek_task(self) -> Optional[Dict[str, Any]]:
        if not self.has_pending_tasks():
            return None
        return self.tasks[self.task_pointer]

    def pop_task(self) -> Optional[Dict[str, Any]]:
        if not self.has_pending_tasks():
            return None
        task = self.tasks[self.task_pointer]
        self.task_pointer += 1
        return task

    def pending_tasks(self) -> List[Dict[str, Any]]:
        return self.tasks[self.task_pointer:]

    def register_tool_call(self, tool_name: str) -> None:
        self.tool_call_counts[tool_name] = int(self.tool_call_counts.get(tool_name, 0)) + 1

    def register_llm_call(self, purpose: str) -> None:
        self.llm_call_counts[purpose] = int(self.llm_call_counts.get(purpose, 0)) + 1

    def tool_call_count(self, tool_name: str) -> int:
        return int(self.tool_call_counts.get(tool_name, 0))

    def llm_call_count(self, purpose: str) -> int:
        return int(self.llm_call_counts.get(purpose, 0))

    def apply_observation(self, observation: Observation) -> None:
        self.observations.append(observation)

        if observation.error:
            self.errors.append(str(observation.error))

        if observation.state_key:
            self.tool_states[observation.state_key] = {
                "status": observation.status,
                "quality": observation.quality,
                "terminal": observation.terminal,
                "tool": observation.tool,
                "summary": observation.summary,
                "data": observation.data,
            }

        if observation.tool == "search_products" and observation.ok:
            self._apply_search_payload(observation.data)

        if observation.tool == "analyze_seller" and observation.ok:
            self._apply_seller_payload(observation.data)

    def _apply_search_payload(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return

        self.search_payload = payload
        results = payload.get("results") or []
        if results and isinstance(results, list):
            first = results[0]
            if isinstance(first, dict):
                self.top_result = first
                seller_name = first.get("seller_name") or first.get("seller_username")
                if seller_name and not self.last_seller_name:
                    self.last_seller_name = str(seller_name)

        analysis = payload.get("analysis")
        if isinstance(analysis, str) and analysis.strip():
            self.search_analysis = analysis.strip()

        metrics = payload.get("metrics") or payload.get("ir_metrics")
        if isinstance(metrics, dict):
            self.metrics = metrics

    def _apply_seller_payload(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return

        self.seller_payload = payload
        seller_name = payload.get("seller_name")
        if seller_name:
            self.last_seller_name = str(seller_name)

    @property
    def search_payload(self) -> Optional[Dict[str, Any]]:
        return getattr(self, "_search_payload", None)

    @search_payload.setter
    def search_payload(self, value: Optional[Dict[str, Any]]) -> None:
        self._search_payload = value

    @property
    def seller_payload(self) -> Optional[Dict[str, Any]]:
        return getattr(self, "_seller_payload", None)

    @seller_payload.setter
    def seller_payload(self, value: Optional[Dict[str, Any]]) -> None:
        self._seller_payload = value

    def has_search_results(self) -> bool:
        payload = self.search_payload or {}
        return int(payload.get("results_count", 0)) > 0 or bool(payload.get("results"))

    def state_status(self, state_key: str) -> Optional[str]:
        state = self.tool_states.get(state_key) or {}
        return state.get("status")

    def has_terminal_state(self, state_key: str) -> bool:
        state = self.tool_states.get(state_key) or {}
        return bool(state.get("terminal"))

    def has_any_terminal_state(self) -> bool:
        return any(bool(state.get("terminal")) for state in self.tool_states.values())

    def terminal_states(self) -> Dict[str, bool]:
        return {
            key: bool(value.get("terminal"))
            for key, value in self.tool_states.items()
        }

    def tool_state_summaries(self) -> Dict[str, Dict[str, Any]]:
        compact: Dict[str, Dict[str, Any]] = {}
        for key, value in self.tool_states.items():
            compact[key] = {
                "tool": value.get("tool"),
                "status": value.get("status"),
                "quality": value.get("quality"),
                "terminal": value.get("terminal"),
                "summary": value.get("summary"),
            }
        return compact

    def recent_observations(self, limit: int = 5) -> List[Dict[str, Any]]:
        items = []
        for obs in self.observations[-limit:]:
            items.append(
                {
                    "tool": obs.tool,
                    "summary": obs.summary,
                    "status": obs.status,
                    "quality": obs.quality,
                }
            )
        return items

    def steps_done(self) -> int:
        return len(self.observations)

    def session_snapshot(self) -> Dict[str, Any]:
        return self.session_memory.snapshot()

    def long_term_snapshot(self) -> Dict[str, Any]:
        return self.long_term_memory.snapshot()

    def scratchpad(self) -> Dict[str, Any]:
        results = (self.search_payload or {}).get("results") or []
        top_results = [_compact_result(item) for item in results[:3] if isinstance(item, dict)]

        seller_summary = None
        if self.seller_payload:
            seller_summary = {
                "seller_name": self.seller_payload.get("seller_name"),
                "count": self.seller_payload.get("count"),
                "trust_score": self.seller_payload.get("trust_score"),
                "sentiment_score": self.seller_payload.get("sentiment_score"),
                "error": self.seller_payload.get("error"),
            }

        return {
            "user_query": self.user_query,
            "steps_done": self.steps_done(),
            "intent": self.detected_intent,
            "pending_tasks": self.pending_tasks(),
            "task_pointer": self.task_pointer,
            "has_search_payload": self.search_payload is not None,
            "has_seller_payload": self.seller_payload is not None,
            "search_status": self.state_status("search"),
            "seller_status": self.state_status("seller"),
            "has_search_results": self.has_search_results(),
            "last_seller_name": self.last_seller_name,
            "top_results": top_results,
            "search_analysis": self.search_analysis,
            "metrics": self.metrics,
            "seller_summary": seller_summary,
            "tool_calls": dict(self.tool_call_counts),
            "llm_calls": dict(self.llm_call_counts),
            "tool_states": self.tool_state_summaries(),
            "recent_observations": self.recent_observations(limit=4),
            "recent_errors": self.errors[-3:],
            "session_memory": self.session_snapshot(),
            "long_term_memory": self.long_term_snapshot(),
        }

    def final_data(self) -> Dict[str, Any]:
        compact_top = _compact_result(self.top_result) if self.top_result else None
        return {
            "intent": self.detected_intent,
            "search": self.search_payload,
            "seller": self.seller_payload,
            "top_result": compact_top,
            "last_seller_name": self.last_seller_name,
            "search_analysis": self.search_analysis,
            "metrics": self.metrics,
            "errors": self.errors[-5:],
            "tool_states": self.tool_state_summaries(),
            "terminal_states": self.terminal_states(),
            "tool_calls": dict(self.tool_call_counts),
            "llm_calls": dict(self.llm_call_counts),
            "pending_tasks": self.pending_tasks(),
            "recent_observations": self.recent_observations(limit=5),
            "session_memory": self.session_snapshot(),
            "long_term_memory": self.long_term_snapshot(),
        }


# Backward-compatible alias used by the rest of the project.
AgentMemory = RequestState