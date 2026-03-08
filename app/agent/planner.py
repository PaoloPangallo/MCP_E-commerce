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

VALID_INTENTS = {"conversation", "seller_analysis", "product_search", "hybrid"}


class ReactPlanner:
    def __init__(self, llm_engine: str = "gemini"):
        self.llm_engine = (llm_engine or "gemini").strip().lower()
        self.max_calls_per_tool = 2

    def decide(
        self,
        memory: AgentMemory,
        step_index: int,
        max_steps: int,
    ) -> PlannerOutput:
        explicit_seller = self._extract_explicit_seller(memory.user_query)
        if explicit_seller and not memory.last_seller_name:
            memory.last_seller_name = explicit_seller

        if memory.has_pending_tasks():
            decision = self._decide_from_task_queue(memory)
            if decision:
                return decision

        llm_decision = self._llm_decide(memory, step_index, max_steps)
        if llm_decision:
            return llm_decision

        return self._safe_fallback_decide(memory)

    def can_stop_early(self, memory: AgentMemory) -> bool:
        if memory.has_pending_tasks():
            return False

        intent = (memory.detected_intent or "").lower()

        if intent == "conversation":
            return True

        if intent == "seller_analysis":
            return memory.has_terminal_state("seller")

        if intent == "product_search":
            return memory.has_terminal_state("search")

        if intent == "hybrid":
            return memory.has_terminal_state("search") and memory.has_terminal_state("seller")

        return False

    def should_abort_after_error(self, memory: AgentMemory, failed_tool: str) -> bool:
        return memory.tool_call_count(failed_tool) >= self.max_calls_per_tool

    def _decide_from_task_queue(self, memory: AgentMemory) -> Optional[PlannerOutput]:
        task = memory.peek_task()
        if not task:
            return None

        tool = task.get("tool")
        if tool not in TOOLS:
            memory.pop_task()
            return None

        if self._exceeds_tool_budget(memory, tool):
            logger.warning("Skipping queued task for tool=%s because budget is exhausted.", tool)
            memory.pop_task()
            return None

        action_input = self._normalize_action_input(tool, task.get("input") or {}, memory)
        if action_input is None:
            logger.warning("Skipping queued task for tool=%s because input normalization failed.", tool)
            memory.pop_task()
            return None

        memory.pop_task()

        return PlannerOutput(
            thought="Eseguo il task pianificato.",
            action=ToolCall(tool=tool, input=action_input),
            intent=self._infer_intent(memory),
        )

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
            logger.info("Planner LLM returned empty output.")
            return None

        json_text = extract_first_json_object(raw)
        if not json_text:
            logger.warning("Planner LLM returned no JSON. Raw head=%r", raw[:180])
            return None

        try:
            payload = json.loads(json_text)
        except Exception as exc:
            logger.warning("Planner LLM returned malformed JSON: %s", exc)
            return None

        thought = str(payload.get("thought") or "").strip()
        intent = str(payload.get("intent") or "").strip().lower()
        action = str(payload.get("action") or "").strip().lower()
        action_input = payload.get("action_input") or {}

        if intent not in VALID_INTENTS:
            logger.info("Planner LLM omitted/invalid intent=%r, inferring from state.", intent)
            intent = self._infer_intent(memory)

        if action in {"finish", "stop"}:
            if memory.has_pending_tasks():
                logger.info("Planner wanted to stop but pending tasks are still present.")
                return self._safe_fallback_decide(memory, forced_intent=intent)

            if intent == "seller_analysis" and not memory.has_terminal_state("seller"):
                return self._safe_fallback_decide(memory, forced_intent=intent)

            if intent == "product_search" and not memory.has_terminal_state("search"):
                return self._safe_fallback_decide(memory, forced_intent=intent)

            if intent == "hybrid" and (
                not memory.has_terminal_state("search")
                or not memory.has_terminal_state("seller")
            ):
                return self._safe_fallback_decide(memory, forced_intent=intent)

            return PlannerOutput(
                thought=thought or "Ho raccolto abbastanza informazioni.",
                should_stop=True,
                intent=intent,
            )

        if action not in TOOLS:
            logger.warning("Planner selected invalid tool=%r.", action)
            return None

        if self._exceeds_tool_budget(memory, action):
            logger.warning("Planner selected tool=%s but budget is exhausted.", action)
            return self._safe_fallback_decide(memory, forced_intent=intent)

        normalized_input = self._normalize_action_input(action, action_input, memory)
        if normalized_input is None:
            logger.warning("Planner selected tool=%s with unusable input=%r.", action, action_input)
            return None

        return PlannerOutput(
            thought=thought or "Procedo con il prossimo step.",
            action=ToolCall(tool=action, input=normalized_input),
            intent=intent,
        )

    def _safe_fallback_decide(
        self,
        memory: AgentMemory,
        forced_intent: Optional[str] = None,
    ) -> PlannerOutput:
        intent = (forced_intent or memory.detected_intent or self._infer_intent(memory)).lower()
        seller = memory.last_seller_name

        if memory.has_pending_tasks():
            decision = self._decide_from_task_queue(memory)
            if decision:
                decision.intent = intent if intent in VALID_INTENTS else decision.intent
                return decision

        if intent == "conversation":
            return PlannerOutput(
                thought="La richiesta è conversazionale.",
                should_stop=True,
                intent="conversation",
            )

        if intent == "seller_analysis":
            if memory.has_terminal_state("seller"):
                return PlannerOutput(
                    thought="Ho già i dati seller necessari.",
                    should_stop=True,
                    intent="seller_analysis",
                )

            if seller and not self._exceeds_tool_budget(memory, "seller_pipeline"):
                return PlannerOutput(
                    thought="Analizzo il venditore richiesto.",
                    action=ToolCall(
                        tool="seller_pipeline",
                        input={"seller_name": seller, "page": 1, "limit": 10},
                    ),
                    intent="seller_analysis",
                )

            return PlannerOutput(
                thought="Manca un venditore esplicito da analizzare.",
                should_stop=True,
                intent="seller_analysis",
                final_answer="Per analizzare il venditore mi serve il suo nome esatto.",
            )

        if intent == "product_search":
            if memory.has_terminal_state("search"):
                return PlannerOutput(
                    thought="Ho già i dati di ricerca necessari.",
                    should_stop=True,
                    intent="product_search",
                )

            if not self._exceeds_tool_budget(memory, "search_pipeline"):
                return PlannerOutput(
                    thought="Cerco i prodotti richiesti.",
                    action=ToolCall(
                        tool="search_pipeline",
                        input={"query": self._clean_search_query(memory.user_query)},
                    ),
                    intent="product_search",
                )

        if intent == "hybrid":
            if not memory.has_terminal_state("seller") and seller and not self._exceeds_tool_budget(memory, "seller_pipeline"):
                return PlannerOutput(
                    thought="Analizzo prima il venditore.",
                    action=ToolCall(
                        tool="seller_pipeline",
                        input={"seller_name": seller, "page": 1, "limit": 10},
                    ),
                    intent="hybrid",
                )

            if not memory.has_terminal_state("search") and not self._exceeds_tool_budget(memory, "search_pipeline"):
                return PlannerOutput(
                    thought="Ora verifico i prodotti.",
                    action=ToolCall(
                        tool="search_pipeline",
                        input={"query": self._clean_search_query(memory.user_query)},
                    ),
                    intent="hybrid",
                )

            return PlannerOutput(
                thought="Ho raccolto tutte le informazioni disponibili.",
                should_stop=True,
                intent="hybrid",
            )

        if memory.has_terminal_state("search") or memory.has_terminal_state("seller"):
            return PlannerOutput(
                thought="Ho già abbastanza informazioni.",
                should_stop=True,
                intent=intent if intent in VALID_INTENTS else None,
            )

        return PlannerOutput(
            thought="Termino.",
            should_stop=True,
            intent=intent if intent in VALID_INTENTS else None,
        )

    def _call_llm(self, prompt: str) -> Optional[str]:
        try:
            if self.llm_engine == "gemini":
                return call_gemini(prompt)

            if self.llm_engine == "ollama":
                return call_ollama(prompt)

        except Exception as exc:
            logger.warning("Planner LLM failed: %s", exc)

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
            query = self._clean_search_query(query)
            if not query:
                return None
            return {"query": query}

        if action == "seller_pipeline":
            seller_name = str(
                clean.get("seller_name")
                or memory.last_seller_name
                or self._extract_explicit_seller(memory.user_query)
                or ""
            ).strip()

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
        q = (query or "").lower()

        q = re.sub(r"(venditore|seller)\s+[a-zA-Z0-9._-]+", "", q)
        q = re.sub(
            r"(dammi i feedback|analizza il venditore|analizza|controlla se vende|verifica se vende|feedback del venditore)",
            "",
            q,
        )
        q = re.sub(r"\s+", " ", q)

        return q.strip()

    def _infer_intent(self, memory: AgentMemory) -> str:
        if memory.tasks:
            tools = {task.get("tool") for task in memory.tasks}
            if {"search_pipeline", "seller_pipeline"} <= tools:
                return "hybrid"
            if "seller_pipeline" in tools:
                return "seller_analysis"
            if "search_pipeline" in tools:
                return "product_search"

        q = (memory.user_query or "").lower()

        greeting_words = ["ciao", "salve", "buongiorno", "hey", "come va"]
        seller_words = ["seller", "venditore", "feedback", "affidabile", "reputazione", "trust"]
        search_words = ["cerca", "trova", "mostra", "vende", "selling", "prodotto", "prodotti", "prezzo"]

        has_greeting = any(word in q for word in greeting_words)
        has_seller = any(word in q for word in seller_words) or bool(self._extract_explicit_seller(q))
        has_search = any(word in q for word in search_words)

        if has_seller and has_search:
            return "hybrid"
        if has_seller:
            return "seller_analysis"
        if has_search:
            return "product_search"
        if has_greeting:
            return "conversation"

        return "conversation"

    def _exceeds_tool_budget(self, memory: AgentMemory, tool_name: str) -> bool:
        return memory.tool_call_count(tool_name) >= self.max_calls_per_tool

    @staticmethod
    def _extract_explicit_seller(text: str) -> Optional[str]:
        pattern = r"(?:venditore|seller)\s+([A-Za-z0-9._-]{3,})"
        match = re.search(pattern, text or "", re.IGNORECASE)
        if match:
            return match.group(1)
        return None
