from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from app.agent.memory import AgentMemory
from app.agent.prompts import build_planner_prompt
from app.agent.schemas import PlannerOutput, ToolCall
from app.agent.tool_registry import TOOLS, get_tool_descriptions
from app.services.parser import call_gemini, call_ollama, extract_first_json_object

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

        explicit_seller = self._extract_explicit_seller(memory.user_query)

        if explicit_seller and not memory.last_seller_name:
            memory.last_seller_name = explicit_seller

        llm_decision = self._llm_decide(memory, step_index, max_steps)

        if llm_decision:
            return llm_decision

        return self._safe_fallback_decide(memory)

    # --------------------------------------------------

    def can_stop_early(self, memory: AgentMemory) -> bool:

        intent = (memory.detected_intent or "").lower()

        if intent == "conversation":
            return True

        if intent == "seller_analysis":
            return memory.seller_payload is not None

        if intent == "product_search":
            return memory.search_payload is not None

        if intent == "hybrid":
            return (
                memory.search_payload is not None
                and memory.seller_payload is not None
            )

        return False

    # --------------------------------------------------

    def _llm_decide(
        self,
        memory: AgentMemory,
        step_index: int,
        max_steps: int,
    ) -> Optional[PlannerOutput]:

        if self.llm_engine == "rule_based":
            return None

        prompt = build_planner_prompt(
            user_query=memory.user_query,
            scratchpad=memory.scratchpad(),
            step_index=step_index,
            max_steps=max_steps,
            tool_descriptions=get_tool_descriptions(),
        )

        raw = self._call_llm(prompt)

        if not raw:
            return None

        json_text = extract_first_json_object(raw)

        if not json_text:
            return None

        try:
            payload = json.loads(json_text)
        except Exception:
            return None

        thought = str(payload.get("thought") or "")
        intent = str(payload.get("intent") or "").lower()
        action = str(payload.get("action") or "").lower()
        action_input = payload.get("action_input") or {}

        if intent not in {"conversation", "seller_analysis", "product_search", "hybrid"}:
            intent = self._infer_intent(memory)

        if action in {"finish", "stop"}:

            if intent == "hybrid":
                if memory.search_payload is None or memory.seller_payload is None:
                    return self._safe_fallback_decide(memory, forced_intent="hybrid")

            return PlannerOutput(
                thought=thought,
                should_stop=True,
                intent=intent,
            )

        if action not in TOOLS:
            return None

        normalized_input = self._normalize_action_input(
            action,
            action_input,
            memory,
        )

        if normalized_input is None:
            return None

        return PlannerOutput(
            thought=thought,
            action=ToolCall(tool=action, input=normalized_input),
            intent=intent,
        )

    # --------------------------------------------------

    def _safe_fallback_decide(
        self,
        memory: AgentMemory,
        forced_intent: Optional[str] = None,
    ) -> PlannerOutput:

        intent = (forced_intent or memory.detected_intent or self._infer_intent(memory)).lower()

        seller = memory.last_seller_name

        task = memory.next_task()

        if task:

            tool = task.get("tool")
            tool_input = task.get("input") or {}

            if tool in TOOLS:
                return PlannerOutput(
                    thought="Eseguo task pianificato.",
                    action=ToolCall(
                        tool=tool,
                        input=tool_input,
                    ),
                )

        if intent == "conversation":

            return PlannerOutput(
                thought="La richiesta è conversazionale.",
                should_stop=True,
                intent="conversation",
            )

        if intent == "seller_analysis":

            if memory.seller_payload is None and seller:

                return PlannerOutput(
                    thought="Analizzo il venditore richiesto.",
                    action=ToolCall(
                        tool="seller_pipeline",
                        input={
                            "seller_name": seller,
                            "page": 1,
                            "limit": 10,
                        },
                    ),
                    intent="seller_analysis",
                )

            return PlannerOutput(
                thought="Ho abbastanza informazioni sul venditore.",
                should_stop=True,
                intent="seller_analysis",
            )

        if intent == "product_search":

            if memory.search_payload is None:

                return PlannerOutput(
                    thought="Cerco i prodotti.",
                    action=ToolCall(
                        tool="search_pipeline",
                        input={
                            "query": self._clean_search_query(memory.user_query),
                        },
                    ),
                    intent="product_search",
                )

            return PlannerOutput(
                thought="Ho abbastanza informazioni sulla ricerca.",
                should_stop=True,
                intent="product_search",
            )

        if intent == "hybrid":

            if memory.seller_payload is None and seller:

                return PlannerOutput(
                    thought="Analizzo prima il venditore.",
                    action=ToolCall(
                        tool="seller_pipeline",
                        input={
                            "seller_name": seller,
                            "page": 1,
                            "limit": 10,
                        },
                    ),
                    intent="hybrid",
                )

            if memory.search_payload is None:

                return PlannerOutput(
                    thought="Ora verifico i prodotti.",
                    action=ToolCall(
                        tool="search_pipeline",
                        input={
                            "query": self._clean_search_query(memory.user_query),
                        },
                    ),
                    intent="hybrid",
                )

            return PlannerOutput(
                thought="Ho raccolto tutte le informazioni.",
                should_stop=True,
                intent="hybrid",
            )

        return PlannerOutput(
            thought="Termino.",
            should_stop=True,
        )

    # --------------------------------------------------

    def _call_llm(self, prompt: str) -> Optional[str]:

        try:

            if self.llm_engine == "gemini":
                return call_gemini(prompt)

            if self.llm_engine == "ollama":
                return call_ollama(prompt)

        except Exception as e:
            logger.warning("Planner LLM failed: %s", e)

        return None

    # --------------------------------------------------

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

            query = self._clean_search_query(query)

            if not query:
                return None

            return {"query": query}

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

    def _clean_search_query(self, query: str) -> str:

        q = query.lower()

        q = re.sub(r"(venditore|seller)\s+[a-zA-Z0-9._-]+", "", q)

        q = re.sub(
            r"(dammi i feedback|analizza|controlla se vende|verifica se vende)",
            "",
            q,
        )

        return q.strip()

    # --------------------------------------------------

    def _infer_intent(self, memory: AgentMemory) -> str:

        q = memory.user_query.lower()

        seller_words = ["seller", "venditore", "feedback"]

        product_words = [
            "compra",
            "vende",
            "selling",
            "search",
            "prezzo",
            "carte",
            "magic",
            "pokemon",
        ]

        has_seller = any(w in q for w in seller_words)
        has_product = any(w in q for w in product_words)

        if has_seller and has_product:
            return "hybrid"

        if has_seller:
            return "seller_analysis"

        if has_product:
            return "product_search"

        return "conversation"

    # --------------------------------------------------

    @staticmethod
    def _extract_explicit_seller(text: str) -> Optional[str]:

        pattern = r"(?:venditore|seller)\s+([A-Za-z0-9._-]{3,})"

        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            return match.group(1)

        return None