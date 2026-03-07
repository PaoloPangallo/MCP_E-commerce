from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from app.agent.memory import AgentMemory
from app.agent.prompts import build_planner_prompt
from app.agent.schemas import PlannerOutput, ToolCall
from app.agent.tool_registry import TOOLS, get_tool_descriptions
from app.services.parser import (
    call_gemini,
    call_ollama,
    extract_first_json_object,
)

logger = logging.getLogger(__name__)


class ReactPlanner:
    def __init__(self, llm_engine: str = "gemini"):
        self.llm_engine = (llm_engine or "gemini").strip().lower()
        self._decision_cache: Dict[str, PlannerOutput] = {}

    def decide(
        self,
        memory: AgentMemory,
        step_index: int,
        max_steps: int,
    ) -> PlannerOutput:
        if step_index > max_steps:
            return PlannerOutput(
                thought="Ho raggiunto il limite di step.",
                should_stop=True,
            )

        # --------------------------------------------------
        # FAST HEURISTIC PATH
        # --------------------------------------------------
        heuristic_first = self._heuristic_fast_path(memory, step_index)
        if heuristic_first is not None:
            return heuristic_first

        llm_decision = self._llm_decide(
            memory=memory,
            step_index=step_index,
            max_steps=max_steps,
        )

        if llm_decision is not None:
            return llm_decision

        return self._heuristic_decide(memory)

    def can_stop_early(self, memory: AgentMemory) -> bool:
        """
        Se l'utente non ha chiesto esplicitamente seller/trust/feedback
        e abbiamo già risultati search, possiamo chiudere.
        """
        query = (memory.user_query or "").lower()

        wants_seller_check = any(
            token in query
            for token in [
                "venditore",
                "seller",
                "affidabile",
                "feedback",
                "fidato",
                "trust",
                "sicuro",
            ]
        )

        return memory.search_payload is not None and not wants_seller_check

    def _heuristic_fast_path(
        self,
        memory: AgentMemory,
        step_index: int,
    ) -> Optional[PlannerOutput]:
        """
        Evita la planner LLM quando il caso è banale:
        - primo step: search quasi sempre
        - secondo step: seller solo se richiesto
        """
        query = (memory.user_query or "").lower()
        explicit_seller = self._extract_explicit_seller(memory.user_query)

        wants_seller_check = any(
            token in query
            for token in [
                "venditore",
                "seller",
                "affidabile",
                "feedback",
                "fidato",
                "trust",
                "sicuro",
            ]
        )

        if step_index == 1 and memory.search_payload is None and memory.seller_payload is None:
            if wants_seller_check and explicit_seller:
                return PlannerOutput(
                    thought="Analizzo direttamente il venditore richiesto.",
                    action=ToolCall(
                        tool="seller_pipeline",
                        input={"seller_name": explicit_seller, "page": 1, "limit": 10},
                    ),
                )

            return PlannerOutput(
                thought="Cerco prima i prodotti più rilevanti.",
                action=ToolCall(
                    tool="search_pipeline",
                    input={"query": memory.user_query},
                ),
            )

        if step_index >= 2:
            if wants_seller_check and memory.search_payload is not None and memory.seller_payload is None and memory.last_seller_name:
                return PlannerOutput(
                    thought="Approfondisco il venditore del risultato migliore.",
                    action=ToolCall(
                        tool="seller_pipeline",
                        input={"seller_name": memory.last_seller_name, "page": 1, "limit": 10},
                    ),
                )

            if memory.search_payload is not None:
                return PlannerOutput(
                    thought="Ho abbastanza elementi per rispondere.",
                    should_stop=True,
                )

        return None

    def _llm_decide(
        self,
        memory: AgentMemory,
        step_index: int,
        max_steps: int,
    ) -> Optional[PlannerOutput]:
        if self.llm_engine == "rule_based":
            return None

        cache_key = self._build_cache_key(memory, step_index, max_steps)
        cached = self._decision_cache.get(cache_key)
        if cached is not None:
            return cached

        prompt = build_planner_prompt(
            user_query=memory.user_query,
            scratchpad=memory.scratchpad(),
            step_index=step_index,
            max_steps=max_steps,
            tool_descriptions=get_tool_descriptions(),
        )

        raw = self._call_planner_llm(prompt)
        if not raw:
            return None

        json_text = extract_first_json_object(raw)
        if not json_text:
            logger.warning("Planner LLM did not return JSON")
            return None

        try:
            payload = json.loads(json_text)
        except Exception as e:
            logger.warning("Planner JSON parse failed: %s", e)
            return None

        if not isinstance(payload, dict):
            return None

        thought = str(payload.get("thought") or "").strip()
        action = str(payload.get("action") or "").strip().lower()
        action_input = payload.get("action_input") or {}
        final_answer = payload.get("final_answer")

        if action in {"finish", "final", "stop", "done"}:
            out = PlannerOutput(
                thought=thought or "Ho abbastanza informazioni per rispondere.",
                should_stop=True,
                final_answer=str(final_answer).strip() if isinstance(final_answer, str) and final_answer.strip() else None,
            )
            self._decision_cache[cache_key] = out
            return out

        if action not in TOOLS:
            logger.warning("Planner returned unknown action: %s", action)
            return None

        normalized_input = self._normalize_action_input(
            action=action,
            action_input=action_input,
            memory=memory,
        )

        if normalized_input is None:
            return None

        if action == "search_pipeline" and memory.search_payload is not None:
            return PlannerOutput(
                thought="Ho già risultati di ricerca sufficienti.",
                should_stop=True,
            )

        if action == "seller_pipeline" and memory.seller_payload is not None:
            current_seller = str(normalized_input.get("seller_name") or "").strip().lower()
            previous_seller = str(memory.seller_payload.get("seller_name") or "").strip().lower()

            if current_seller and previous_seller and current_seller == previous_seller:
                return PlannerOutput(
                    thought="Ho già analizzato questo venditore.",
                    should_stop=True,
                )

        out = PlannerOutput(
            thought=thought or "Procedo con il tool più utile.",
            action=ToolCall(tool=action, input=normalized_input),
        )
        self._decision_cache[cache_key] = out
        return out

    def _call_planner_llm(self, prompt: str) -> Optional[str]:
        try:
            if self.llm_engine == "gemini":
                return call_gemini(prompt)

            if self.llm_engine == "ollama":
                return call_ollama(prompt)

            return None
        except Exception as e:
            logger.warning("Planner LLM call failed: %s", e)
            return None

    def _normalize_action_input(
        self,
        action: str,
        action_input: Dict[str, Any],
        memory: AgentMemory,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(action_input, dict):
            action_input = {}

        clean = dict(action_input)

        if action == "search_pipeline":
            query = str(clean.get("query") or memory.user_query).strip()
            if not query:
                return None

            clean = {"query": query}
            return clean

        if action == "seller_pipeline":
            seller_name = str(
                clean.get("seller_name")
                or memory.last_seller_name
                or ""
            ).strip()

            if not seller_name:
                seller_name = self._extract_explicit_seller(memory.user_query) or ""

            if not seller_name:
                return None

            try:
                page = max(1, int(clean.get("page", 1)))
            except Exception:
                page = 1

            try:
                limit = min(max(int(clean.get("limit", 10)), 1), 50)
            except Exception:
                limit = 10

            return {
                "seller_name": seller_name,
                "page": page,
                "limit": limit,
            }

        return None

    def _heuristic_decide(self, memory: AgentMemory) -> PlannerOutput:
        query = (memory.user_query or "").lower()

        wants_seller_check = any(
            token in query
            for token in [
                "venditore",
                "seller",
                "affidabile",
                "feedback",
                "fidato",
                "trust",
                "sicuro",
            ]
        )

        explicit_seller = self._extract_explicit_seller(memory.user_query)

        if memory.search_payload is None and memory.seller_payload is None:
            if wants_seller_check and explicit_seller:
                return PlannerOutput(
                    thought="Analizzo direttamente il venditore richiesto.",
                    action=ToolCall(
                        tool="seller_pipeline",
                        input={"seller_name": explicit_seller, "page": 1, "limit": 10},
                    ),
                )

            return PlannerOutput(
                thought="Cerco prima i prodotti più rilevanti.",
                action=ToolCall(
                    tool="search_pipeline",
                    input={"query": memory.user_query},
                ),
            )

        if wants_seller_check and memory.seller_payload is None and memory.last_seller_name:
            return PlannerOutput(
                thought="Approfondisco il venditore del risultato migliore.",
                action=ToolCall(
                    tool="seller_pipeline",
                    input={"seller_name": memory.last_seller_name, "page": 1, "limit": 10},
                ),
            )

        return PlannerOutput(
            thought="Ho abbastanza elementi per rispondere.",
            should_stop=True,
        )

    @staticmethod
    def _extract_explicit_seller(text: str) -> Optional[str]:
        if not text:
            return None

        patterns = [
            r"(?:venditore|seller)\s+([A-Za-z0-9._-]{3,})",
            r'"([A-Za-z0-9._-]{3,})"',
            r"'([A-Za-z0-9._-]{3,})'",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                seller_name = match.group(1).strip()
                if seller_name:
                    return seller_name

        return None

    @staticmethod
    def _build_cache_key(
        memory: AgentMemory,
        step_index: int,
        max_steps: int,
    ) -> str:
        return json.dumps(
            {
                "q": memory.user_query,
                "scratchpad": memory.scratchpad(),
                "step": step_index,
                "max_steps": max_steps,
            },
            ensure_ascii=False,
            sort_keys=True,
        )