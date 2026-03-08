
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from app.agent.memory import AgentMemory
from app.agent.prompts import build_planner_prompt
from app.agent.schemas import PlannerOutput, ToolCall
from app.agent.tool_registry import (
    TOOLS,
    extract_explicit_seller,
    find_first_tool_by_tags,
    get_tool_catalog,
    get_tool_spec,
)
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
        explicit_seller = extract_explicit_seller(memory.user_query)
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

        return self._intent_is_satisfied(memory, intent)

    def should_abort_after_error(self, memory: AgentMemory, failed_tool: str) -> bool:
        return memory.tool_call_count(failed_tool) >= self.max_calls_per_tool

    def _decide_from_task_queue(self, memory: AgentMemory) -> Optional[PlannerOutput]:
        task = memory.peek_task()
        if not task:
            return None

        tool = str(task.get("tool") or "").strip()
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
            tool_catalog=get_tool_catalog(),
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

            if not self._intent_is_satisfied(memory, intent):
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

        for tool_name in self._ordered_tools_for_intent(intent):
            if self._tool_state_is_terminal(memory, tool_name):
                continue

            if self._exceeds_tool_budget(memory, tool_name):
                continue

            normalized_input = self._normalize_action_input(tool_name, {}, memory)
            if normalized_input is None:
                continue

            return PlannerOutput(
                thought=f"Uso il tool più adatto: {tool_name}.",
                action=ToolCall(tool=tool_name, input=normalized_input),
                intent=intent if intent in VALID_INTENTS else self._infer_intent(memory),
            )

        if memory.has_any_terminal_state():
            return PlannerOutput(
                thought="Ho già abbastanza informazioni.",
                should_stop=True,
                intent=intent if intent in VALID_INTENTS else self._infer_intent(memory),
            )

        if intent == "seller_analysis":
            return PlannerOutput(
                thought="Manca un venditore esplicito da analizzare.",
                should_stop=True,
                intent="seller_analysis",
                final_answer="Per analizzare il venditore mi serve il suo nome esatto.",
            )

        return PlannerOutput(
            thought="Termino.",
            should_stop=True,
            intent=intent if intent in VALID_INTENTS else self._infer_intent(memory),
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
        spec = get_tool_spec(action)
        if spec is None:
            return None

        clean = dict(action_input or {})

        try:
            if spec.input_normalizer:
                clean = spec.input_normalizer(clean, memory)
        except Exception:
            return None

        for field_name in spec.required_fields:
            value = clean.get(field_name)
            if value is None or (isinstance(value, str) and not value.strip()):
                return None

        return clean

    def _intent_is_satisfied(self, memory: AgentMemory, intent: str) -> bool:
        if intent == "conversation":
            return True

        tools = self._ordered_tools_for_intent(intent)
        if not tools:
            return memory.has_any_terminal_state()

        if intent == "hybrid":
            return all(self._tool_state_is_terminal(memory, tool_name) for tool_name in tools)

        return any(self._tool_state_is_terminal(memory, tool_name) for tool_name in tools)

    def _ordered_tools_for_intent(self, intent: str) -> list[str]:
        seller_tool = find_first_tool_by_tags("seller", "trust", "feedback")
        search_tool = find_first_tool_by_tags("search", "product", "catalog")

        if intent == "seller_analysis":
            return [tool for tool in [seller_tool] if tool]

        if intent == "product_search":
            return [tool for tool in [search_tool] if tool]

        if intent == "hybrid":
            ordered = [tool for tool in [seller_tool, search_tool] if tool]
            seen: set[str] = set()
            unique: list[str] = []
            for tool in ordered:
                if tool not in seen:
                    seen.add(tool)
                    unique.append(tool)
            return unique

        return []

    def _tool_state_is_terminal(self, memory: AgentMemory, tool_name: str) -> bool:
        spec = get_tool_spec(tool_name)
        if spec is None or not spec.state_key:
            return False
        return memory.has_terminal_state(spec.state_key)

    def _tool_matches_any_tag(self, tool_name: str, tags: set[str]) -> bool:
        spec = get_tool_spec(tool_name)
        if spec is None:
            return False
        return bool({tag.lower() for tag in spec.tags} & tags)

    def _infer_intent(self, memory: AgentMemory) -> str:
        if memory.tasks:
            tools = [str(task.get("tool") or "").strip() for task in memory.tasks]
            has_seller = any(self._tool_matches_any_tag(tool, {"seller", "trust", "feedback"}) for tool in tools)
            has_search = any(self._tool_matches_any_tag(tool, {"search", "product", "catalog"}) for tool in tools)

            if has_search and has_seller:
                return "hybrid"
            if has_seller:
                return "seller_analysis"
            if has_search:
                return "product_search"

        q = (memory.user_query or "").lower()

        greeting_words = ["ciao", "salve", "buongiorno", "hey", "come va", "come stai", "tutto bene"]
        seller_words = ["seller", "venditore", "feedback", "affidabile", "reputazione", "trust", "recensioni"]
        search_words = ["cerca", "trova", "mostra", "vende", "selling", "prodotto", "prodotti", "prezzo", "compra"]

        has_greeting = any(word in q for word in greeting_words)
        has_seller = any(word in q for word in seller_words) or bool(extract_explicit_seller(q))
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
