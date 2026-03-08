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

    search_payload: Optional[Dict[str, Any]] = None
    seller_payload: Optional[Dict[str, Any]] = None
    tasks: List[Dict] = field(default_factory=list)
    task_pointer: int = 0

    top_result: Optional[Dict[str, Any]] = None
    last_seller_name: Optional[str] = None
    search_analysis: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

    detected_intent: Optional[str] = None
    errors: List[str] = field(default_factory=list)

    def load_tasks(self, tasks: List[Dict]):

        if tasks:
            self.tasks = tasks
            self.task_pointer = 0

    def next_task(self):

        if self.task_pointer >= len(self.tasks):
            return None

        task = self.tasks[self.task_pointer]
        self.task_pointer += 1

        return task

    def apply_observation(self, observation: Observation) -> None:
        self.observations.append(observation)

        if not observation.ok:
            if observation.error:
                self.errors.append(str(observation.error))
            return

        data = observation.data or {}

        if observation.tool == "search_pipeline":
            self._apply_search_payload(data)
            return

        if observation.tool == "seller_pipeline":
            self._apply_seller_payload(data)
            return

    def _apply_search_payload(self, data: Dict[str, Any]) -> None:
        self.search_payload = data
        self.search_analysis = data.get("analysis")
        self.metrics = data.get("metrics") or data.get("ir_metrics")

        results = data.get("results") or []
        self.top_result = results[0] if results else None

        if self.top_result:
            seller_name = (
                self.top_result.get("seller_name")
                or self.top_result.get("seller_username")
            )
            if seller_name:
                self.last_seller_name = str(seller_name).strip()

    def _apply_seller_payload(self, data: Dict[str, Any]) -> None:
        self.seller_payload = data

        seller_name = data.get("seller_name")
        if seller_name:
            self.last_seller_name = str(seller_name).strip()

        error = data.get("error")
        if error:
            self.errors.append(str(error))

    def has_search_results(self) -> bool:
        if not self.search_payload:
            return False
        results = self.search_payload.get("results") or []
        return len(results) > 0

    def has_seller_data(self) -> bool:
        return self.seller_payload is not None

    def has_seller_error(self) -> bool:
        if not self.seller_payload:
            return False
        return bool(self.seller_payload.get("error"))

    def steps_done(self) -> int:
        return len(self.observations)

    def scratchpad(self) -> Dict[str, Any]:
        results = (self.search_payload or {}).get("results") or []
        top_results = [_compact_result(item) for item in results[:3]]

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
            "has_search_payload": self.search_payload is not None,
            "has_seller_payload": self.seller_payload is not None,
            "has_search_results": self.has_search_results(),
            "last_seller_name": self.last_seller_name,
            "top_results": top_results,
            "search_analysis": self.search_analysis,
            "metrics": self.metrics,
            "seller_summary": seller_summary,
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
        }