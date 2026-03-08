
from __future__ import annotations

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


@dataclass
class AgentMemory:
    user_query: str
    observations: List[Observation] = field(default_factory=list)

    tasks: List[Dict[str, Any]] = field(default_factory=list)
    task_pointer: int = 0

    detected_intent: Optional[str] = None
    errors: List[str] = field(default_factory=list)

    tool_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tool_call_counts: Dict[str, int] = field(default_factory=dict)

    top_result: Optional[Dict[str, Any]] = None
    last_seller_name: Optional[str] = None
    search_analysis: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

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

        if cleaned:
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
        if not self.has_pending_tasks():
            return []
        return self.tasks[self.task_pointer:]

    def register_tool_call(self, tool_name: str) -> None:
        self.tool_call_counts[tool_name] = self.tool_call_counts.get(tool_name, 0) + 1

    def tool_call_count(self, tool_name: str) -> int:
        return self.tool_call_counts.get(tool_name, 0)

    def apply_observation(self, observation: Observation) -> None:
        self.observations.append(observation)

        if observation.error:
            self.errors.append(str(observation.error))

        state_key = observation.state_key
        payload = observation.state_update or observation.data or {}

        if state_key:
            self.tool_states[state_key] = {
                "status": observation.status,
                "quality": observation.quality,
                "terminal": observation.terminal,
                "payload": payload,
                "tool": observation.tool,
                "attempts": payload.get("_tool_attempts", 1),
                "summary": observation.summary,
            }

        self._refresh_derived_fields(payload)

    def _refresh_derived_fields(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return

        analysis = payload.get("analysis")
        if analysis:
            self.search_analysis = str(analysis)

        metrics = payload.get("metrics") or payload.get("ir_metrics")
        if metrics:
            self.metrics = metrics

        seller_name = payload.get("seller_name")
        if seller_name:
            self.last_seller_name = str(seller_name).strip()

        results = payload.get("results") or []
        if isinstance(results, list) and results:
            top = results[0]
            if isinstance(top, dict):
                self.top_result = top
                top_seller = top.get("seller_name") or top.get("seller_username")
                if top_seller:
                    self.last_seller_name = str(top_seller).strip()

    def state(self, state_key: str) -> Optional[Dict[str, Any]]:
        return self.tool_states.get(state_key)

    def payload(self, state_key: str) -> Optional[Dict[str, Any]]:
        state = self.state(state_key)
        return state.get("payload") if state else None

    def has_terminal_state(self, state_key: str) -> bool:
        state = self.state(state_key)
        return bool(state and state.get("terminal"))

    def has_any_terminal_state(self) -> bool:
        return any(bool(state.get("terminal")) for state in self.tool_states.values())

    def has_useful_state(self, state_key: str) -> bool:
        state = self.state(state_key)
        if not state:
            return False

        quality = state.get("quality")
        return quality in {"partial", "good"}

    def state_status(self, state_key: str) -> Optional[str]:
        state = self.state(state_key)
        return state.get("status") if state else None

    def terminal_states(self) -> Dict[str, bool]:
        return {
            key: bool(value.get("terminal"))
            for key, value in self.tool_states.items()
        }

    def tool_state_summaries(self) -> Dict[str, Dict[str, Any]]:
        return {
            key: {
                "tool": value.get("tool"),
                "status": value.get("status"),
                "quality": value.get("quality"),
                "terminal": value.get("terminal"),
                "attempts": value.get("attempts"),
                "summary": value.get("summary"),
            }
            for key, value in self.tool_states.items()
        }

    @property
    def search_payload(self) -> Optional[Dict[str, Any]]:
        return self.payload("search")

    @property
    def seller_payload(self) -> Optional[Dict[str, Any]]:
        return self.payload("seller")

    def has_search_results(self) -> bool:
        results = (self.search_payload or {}).get("results") or []
        return len(results) > 0

    def has_seller_data(self) -> bool:
        status = self.state_status("seller")
        return status in {"ok", "no_data"}

    def has_seller_error(self) -> bool:
        return bool((self.seller_payload or {}).get("error"))

    def steps_done(self) -> int:
        return len(self.observations)

    def recent_observations(self, limit: int = 5) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []

        for observation in self.observations[-limit:]:
            items.append(
                {
                    "tool": observation.tool,
                    "status": observation.status,
                    "quality": observation.quality,
                    "summary": observation.summary,
                    "state_key": observation.state_key,
                    "terminal": observation.terminal,
                }
            )

        return items

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
            "tool_states": self.tool_state_summaries(),
            "recent_observations": self.recent_observations(limit=4),
            "recent_errors": self.errors[-3:],
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
            "tool_calls": self.tool_call_counts,
            "pending_tasks": self.pending_tasks(),
            "recent_observations": self.recent_observations(limit=5),
        }
