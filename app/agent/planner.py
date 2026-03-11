from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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

VALID_INTENTS = {"conversation", "seller_analysis", "product_search", "hybrid", "comparison"}

QUESTION_WORDS = {
    "chi", "cosa", "come", "quando", "dove", "quale", "quali", "perché", "perche",
}

CONVERSATION_CUES = {
    "penso", "pensi", "pensate", "credo", "credi", "sapere", "sai", "spiegami",
    "raccontami", "parliamo", "dimmi", "opinione", "consiglio", "aiutami",
    "aiuto", "discutiamo", "intendo", "significa", "vuol", "dire", "bro",
}

GREETING_CUES = {
    "ciao", "salve", "buongiorno", "buonasera", "hey", "ehi", "hello",
}

SELLER_CUES = {
    "seller", "venditore", "negozio", "shop", "feedback", "affidabile",
    "reputazione", "trust", "recensioni", "serio", "sicuro",
}

TRANSACTIONAL_CUES = {
    "cerca", "cerco", "trova", "trovami", "mostra", "vorrei", "voglio", "mi serve",
    "compra", "acquistare", "prezzo", "prezzi", "costo", "budget", "massimo",
    "minimo", "sotto", "meno", "entro", "offerta", "offerte",
}

COMPARISON_CUES = {
    "compara", "compari", "confronta", "confronto", "differenza", "differenze",
    "meglio", "peggio", "versus", "vs", "comparazione",
}

ATTRIBUTE_CUES = {
    "taglia", "numero", "misura", "colore", "materiale", "marca", "modello",
    "uomo", "donna", "bambino", "bambina", "adulto", "adulti", "nuovo", "usato",
    "con", "senza", "zip", "cappuccio", "manica", "maniche",
    "nero", "nera", "bianco", "bianca", "blu", "rosso", "rossa",
    "verde", "grigio", "grigia", "beige", "m", "l", "xl", "xxl", "xs", "s",
}

PERSONAL_PRONOUNS = {
    "io", "tu", "noi", "voi", "me", "te", "mi", "ti", "mio", "mia", "tuo", "tua",
}

VERBISH_CUES = {
    "sei", "sono", "è", "e", "stai", "va", "pensi", "pensa", "credi", "crede",
    "sai", "sapete", "fai", "fare", "fate", "posso", "puoi", "potresti",
}

MODEL_CODE_RE = re.compile(r"\b[a-z]{1,10}[\-_\s]?\d{1,5}[a-z0-9\-_]*\b", re.IGNORECASE)
PRICE_RE = re.compile(r"\b\d{1,5}(?:[.,]\d{1,2})?\s*(?:€|euro)\b", re.IGNORECASE)
SIZE_RE = re.compile(
    r"\b(?:taglia|misura|numero|eu|uk|us|it)\s*[a-z]?\d{1,3}\b|\b(?:xs|s|m|l|xl|xxl)\b",
    re.IGNORECASE,
)
ALT_SIZE_RE = re.compile(r"\b\d{1,3}\s*(?:o|oppure|/|-)\s*\d{1,3}\b", re.IGNORECASE)
PRICE_BOUND_RE = re.compile(r"\b(?:max|massimo|minimo|budget|entro|sotto|meno di|al massimo)\b", re.IGNORECASE)
TOKEN_RE = re.compile(r"[\wÀ-ÿ]+", re.UNICODE)


@dataclass
class IntentEvidence:
    product: float = 0.0
    seller: float = 0.0
    conversation: float = 0.0
    comparison: float = 0.0
    reasons: Dict[str, list[str]] = field(
        default_factory=lambda: {"product": [], "seller": [], "conversation": [], "comparison": []}
    )

    def add(self, label: str, value: float, reason: str) -> None:
        if label == "product":
            self.product += value
        elif label == "seller":
            self.seller += value
        elif label == "conversation":
            self.conversation += value
        elif label == "comparison":
            self.comparison += value
        if reason:
            self.reasons.setdefault(label, []).append(reason)

    def top_two(self) -> tuple[tuple[str, float], tuple[str, float]]:
        ordered = sorted(
            [
                ("product_search", self.product),
                ("seller_analysis", self.seller),
                ("conversation", self.conversation),
                ("comparison", self.comparison),
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        return ordered[0], ordered[1]


class ReactPlanner:
    """
    Planner orientato ai capability-tool.
    In Fase 1 manteniamo il cuore attuale, ma:
    - usiamo i nuovi nomi tool (`search_products`, `analyze_seller`, `conversation`)
    - aggiungiamo uno strato state-based prima del routing più astratto
    """

    def __init__(self, llm_engine: str = "gemini"):
        self.llm_engine = (llm_engine or "gemini").strip().lower()
        self.max_calls_per_tool = 2
        self.intent_threshold = 0.55
        self.margin_threshold = 0.18
        self.hybrid_threshold = 0.62

    async def decide(
            self,
            memory: AgentMemory,
            step_index: int,
            max_steps: int,
            custom_instructions: Optional[str] = None
    ) -> PlannerOutput:
        explicit_seller = extract_explicit_seller(memory.user_query)
        if explicit_seller and not memory.last_seller_name:
            memory.last_seller_name = explicit_seller

        if memory.has_pending_tasks():
            decision = self._decide_from_task_queue(memory)
            if decision:
                return decision

        # FIX B:
        # fast path conversazionale prima di qualunque possibile chiamata lenta al planner LLM
        conversation_fast_path = self._conversation_fast_path(memory)
        if conversation_fast_path:
            return conversation_fast_path

        deterministic = self._deterministic_decide(memory)
        if deterministic:
            return deterministic

        llm_decision = await self._llm_decide(memory, step_index, max_steps, custom_instructions=custom_instructions)
        if llm_decision:
            return llm_decision

        return self._safe_fallback_decide(memory)

    def _conversation_fast_path(self, memory: AgentMemory) -> Optional[PlannerOutput]:

        text = (memory.user_query or "").strip().lower()

        if text in {"ciao", "hey", "hello", "salve"}:
            return PlannerOutput(
                thought="Saluto semplice.",
                should_stop=True,
                intent="conversation",
            )

        intent, confidence, evidence = self._infer_intent_with_confidence(memory)

        if intent != "conversation":
            return None

        # alta confidenza conversazionale
        if confidence >= self.intent_threshold:
            return PlannerOutput(
                thought="La richiesta è chiaramente conversazionale.",
                should_stop=True,
                intent="conversation",
            )

        # anche con confidenza media, se non ci sono segnali seller/search,
        # evitiamo di mandare una banalità a Ollama solo per decidere.
        if (
                evidence.conversation >= 0.40
                and evidence.product < 0.30
                and evidence.seller < 0.30
        ):
            return PlannerOutput(
                thought="La richiesta sembra conversazionale.",
                should_stop=True,
                intent="conversation",
            )

        return None

    def can_stop_early(self, memory: AgentMemory) -> bool:
        if memory.has_pending_tasks():
            return False

        intent = (memory.detected_intent or "").lower()
        if intent == "conversation":
            return True

        return self._intent_is_satisfied(memory, intent)

    def should_abort_after_error(self, memory: AgentMemory, failed_tool: str) -> bool:
        return memory.tool_call_count(failed_tool) >= self.max_calls_per_tool

    def _state_based_decide(self, memory: AgentMemory) -> Optional[PlannerOutput]:
        intent = (memory.detected_intent or self._infer_intent(memory)).lower()

        search_tool = self._search_tool_name()
        seller_tool = self._seller_tool_name()
        compare_tool = self._compare_tool_name()

        if intent == "conversation":
            return PlannerOutput(
                thought="La richiesta è conversazionale.",
                should_stop=True,
                intent="conversation",
            )

        if intent == "seller_analysis":
            if seller_tool and not self._tool_state_is_terminal(memory, seller_tool):
                normalized = self._normalize_action_input(seller_tool, {}, memory)
                if normalized is not None and not self._exceeds_tool_budget(memory, seller_tool):
                    return PlannerOutput(
                        thought="Devo ancora analizzare il venditore.",
                        action=ToolCall(tool=seller_tool, input=normalized),
                        intent="seller_analysis",
                    )

        if intent == "product_search":
            if search_tool and not self._tool_state_is_terminal(memory, search_tool):
                normalized = self._normalize_action_input(search_tool, {}, memory)
                if normalized is not None and not self._exceeds_tool_budget(memory, search_tool):
                    return PlannerOutput(
                        thought="Devo ancora cercare i prodotti.",
                        action=ToolCall(tool=search_tool, input=normalized),
                        intent="product_search",
                    )

        if intent == "comparison":
            if compare_tool and not self._tool_state_is_terminal(memory, compare_tool):
                normalized = self._normalize_action_input(compare_tool, {}, memory)
                if normalized is not None and not self._exceeds_tool_budget(memory, compare_tool):
                    return PlannerOutput(
                        thought="Devo ancora confrontare i prodotti.",
                        action=ToolCall(tool=compare_tool, input=normalized),
                        intent="comparison",
                    )

        if intent == "hybrid":
            if search_tool and not self._tool_state_is_terminal(memory, search_tool):
                normalized = self._normalize_action_input(search_tool, {}, memory)
                if normalized is not None and not self._exceeds_tool_budget(memory, search_tool):
                    return PlannerOutput(
                        thought="Prima completo la ricerca prodotti.",
                        action=ToolCall(tool=search_tool, input=normalized),
                        intent="hybrid",
                    )

            if seller_tool and not self._tool_state_is_terminal(memory, seller_tool):
                normalized = self._normalize_action_input(seller_tool, {}, memory)
                if normalized is not None and not self._exceeds_tool_budget(memory, seller_tool):
                    return PlannerOutput(
                        thought="Ora controllo il venditore.",
                        action=ToolCall(tool=seller_tool, input=normalized),
                        intent="hybrid",
                    )

        return None

    def _deterministic_decide(self, memory: AgentMemory) -> Optional[PlannerOutput]:
        intent, confidence, evidence = self._infer_intent_with_confidence(memory)

        if intent == "conversation" and confidence >= self.intent_threshold:
            return PlannerOutput(
                thought="La richiesta è conversazionale.",
                should_stop=True,
                intent="conversation",
            )

        if intent not in VALID_INTENTS:
            return None

        if confidence < self.intent_threshold:
            return None

        if self._intent_is_satisfied(memory, intent):
            return PlannerOutput(
                thought="Ho già raccolto dati sufficienti.",
                should_stop=True,
                intent=intent,
            )

        for tool_name in self._ordered_tools_for_intent(intent):
            if self._tool_state_is_terminal(memory, tool_name):
                continue
            if self._exceeds_tool_budget(memory, tool_name):
                continue

            normalized_input = self._normalize_action_input(tool_name, {}, memory)
            if normalized_input is None:
                continue

            why = self._reason_summary(intent, evidence)
            return PlannerOutput(
                thought=why or f"Uso il tool più adatto: {tool_name}.",
                action=ToolCall(tool=tool_name, input=normalized_input),
                intent=intent,
            )

        return None

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

    async def _llm_decide(
            self,
            memory: AgentMemory,
            step_index: int,
            max_steps: int,
            custom_instructions: Optional[str] = None
    ) -> Optional[PlannerOutput]:
        if self.llm_engine == "rule_based":
            return None

        prompt = build_planner_prompt(
            user_query=memory.user_query,
            scratchpad=memory.scratchpad(),
            step_index=step_index,
            max_steps=max_steps,
            tool_catalog=get_tool_catalog(),
            custom_instructions=custom_instructions
        )

        raw = await self._call_llm(prompt)
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
            intent = self._infer_intent(memory)

        if action in {"finish", "stop"}:
            if memory.has_pending_tasks():
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

        if self._intent_is_satisfied(memory, intent):
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

    async def _call_llm(self, prompt: str) -> Optional[str]:
        import asyncio
        try:
            if self.llm_engine == "gemini":
                return await asyncio.to_thread(call_gemini, prompt)
            if self.llm_engine == "ollama":
                return await asyncio.to_thread(call_ollama, prompt)
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

    def _search_tool_name(self) -> Optional[str]:
        return find_first_tool_by_tags("search", "product", "catalog")

    def _seller_tool_name(self) -> Optional[str]:
        return find_first_tool_by_tags("seller", "trust", "feedback")

    def _compare_tool_name(self) -> Optional[str]:
        return find_first_tool_by_tags("compare", "product")

    def _ordered_tools_for_intent(self, intent: str) -> list[str]:
        seller_tool = self._seller_tool_name()
        search_tool = self._search_tool_name()
        compare_tool = self._compare_tool_name()

        if intent == "comparison":
            return [tool for tool in [compare_tool] if tool]

        if intent == "seller_analysis":
            return [tool for tool in [seller_tool] if tool]

        if intent == "product_search":
            return [tool for tool in [search_tool] if tool]

        if intent == "hybrid":
            ordered = [tool for tool in [search_tool, seller_tool] if tool]
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
        return self._infer_intent_with_confidence(memory)[0]

    def _infer_intent_with_confidence(self, memory: AgentMemory) -> tuple[str, float, IntentEvidence]:
        if memory.tasks:
            tools = [str(task.get("tool") or "").strip() for task in memory.tasks]
            has_seller = any(self._tool_matches_any_tag(tool, {"seller", "trust", "feedback"}) for tool in tools)
            has_search = any(self._tool_matches_any_tag(tool, {"search", "product", "catalog"}) for tool in tools)

            if has_search and has_seller:
                return "hybrid", 1.0, IntentEvidence(product=1.0, seller=1.0)
            if has_seller:
                return "seller_analysis", 1.0, IntentEvidence(seller=1.0)
            if has_search:
                return "product_search", 1.0, IntentEvidence(product=1.0)

        evidence = self._score_query(memory.user_query or "", memory)
        top, second = evidence.top_two()
        label, score = top
        margin = score - second[1]

        if evidence.product >= self.hybrid_threshold and evidence.seller >= self.hybrid_threshold:
            return "hybrid", min(evidence.product, evidence.seller), evidence

        if score >= self.intent_threshold and margin >= self.margin_threshold:
            return label, score, evidence

        if label == "seller_analysis" and evidence.seller >= self.intent_threshold:
            return label, evidence.seller, evidence
        if label == "conversation" and evidence.conversation >= self.intent_threshold and evidence.product < 0.45:
            return label, evidence.conversation, evidence
        if label == "product_search" and evidence.product >= self.intent_threshold and evidence.conversation < 0.45:
            return label, evidence.product, evidence

        return label, max(score, 0.0), evidence

    def _score_query(self, text: str, memory: AgentMemory) -> IntentEvidence:
        q = (text or "").strip().lower()
        ev = IntentEvidence()

        if not q:
            ev.add("conversation", 0.8, "empty_query")
            return ev

        tokens = [t for t in TOKEN_RE.findall(q) if t]
        token_set = set(tokens)
        token_count = len(tokens)

        explicit_seller = extract_explicit_seller(q)
        if explicit_seller:
            ev.add("seller", 0.9, "explicit_seller")

        seller_hits = len(token_set & SELLER_CUES)
        if seller_hits:
            ev.add("seller", min(0.25 + 0.12 * seller_hits, 0.6), "seller_lexicon")

        transactional_hits = len(token_set & TRANSACTIONAL_CUES)
        if transactional_hits:
            ev.add("product", min(0.18 + 0.08 * transactional_hits, 0.45), "transactional_lexicon")

        comparison_hits = len(token_set & COMPARISON_CUES)
        if comparison_hits:
            ev.add("comparison", min(0.35 + 0.15 * comparison_hits, 0.8), "comparison_lexicon")
            ev.add("product", 0.2, "comparison_implies_product")

        attribute_hits = len(token_set & ATTRIBUTE_CUES)
        if attribute_hits:
            ev.add("product", min(0.12 + 0.06 * attribute_hits, 0.42), "attribute_lexicon")

        greeting_hits = len(token_set & GREETING_CUES)
        if greeting_hits:
            ev.add("conversation", min(0.15 + 0.08 * greeting_hits, 0.28), "greeting")

        conversation_hits = len(token_set & CONVERSATION_CUES)
        if conversation_hits:
            ev.add("conversation", min(0.2 + 0.08 * conversation_hits, 0.5), "conversation_lexicon")

        pronoun_hits = len(token_set & PERSONAL_PRONOUNS)
        if pronoun_hits:
            ev.add("conversation", min(0.08 + 0.04 * pronoun_hits, 0.18), "personal_pronouns")

        if any(tok in QUESTION_WORDS for tok in tokens[:2]):
            ev.add("conversation", 0.18, "question_word_prefix")

        if q.endswith("?"):
            ev.add("conversation", 0.08, "question_mark")

        if PRICE_RE.search(q):
            ev.add("product", 0.34, "price")
        if PRICE_BOUND_RE.search(q):
            ev.add("product", 0.24, "price_bound")
        if SIZE_RE.search(q) or ALT_SIZE_RE.search(q):
            ev.add("product", 0.26, "size_or_variant")
        if MODEL_CODE_RE.search(q):
            ev.add("product", 0.16, "model_code")
            # Multiple model codes strongly imply comparison
            if len(MODEL_CODE_RE.findall(q)) >= 2:
                ev.add("comparison", 0.45, "multiple_model_codes")

        if re.search(r"\b(?:con|senza)\b", q) and attribute_hits:
            ev.add("product", 0.12, "attribute_composition")

        if re.search(r"\b(?:da|per)\s+(?:uomo|donna|bambin[oa]|adult[io])\b", q):
            ev.add("product", 0.22, "demographic_filter")

        has_conversation_structure = bool(token_set & VERBISH_CUES) or any(tok in QUESTION_WORDS for tok in tokens)
        has_product_constraints = bool(
            PRICE_RE.search(q)
            or PRICE_BOUND_RE.search(q)
            or SIZE_RE.search(q)
            or ALT_SIZE_RE.search(q)
            or (token_set & ATTRIBUTE_CUES)
            or (token_set & TRANSACTIONAL_CUES)
        )

        if 2 <= token_count <= 9 and has_product_constraints and not has_conversation_structure:
            ev.add("product", 0.24, "short_keyword_like_query")

        if token_count >= 5 and has_conversation_structure and not has_product_constraints and not explicit_seller:
            ev.add("conversation", 0.26, "open_ended_sentence")

        if explicit_seller and has_product_constraints:
            ev.add("product", 0.18, "seller_plus_product_constraints")

        return ev

    def _reason_summary(self, intent: str, evidence: IntentEvidence) -> str:
        mapping = {
            "product_search": evidence.reasons.get("product", []),
            "seller_analysis": evidence.reasons.get("seller", []),
            "conversation": evidence.reasons.get("conversation", []),
            "comparison": evidence.reasons.get("comparison", []),
            "hybrid": evidence.reasons.get("seller", []) + evidence.reasons.get("product", []),
        }
        reasons = [r for r in mapping.get(intent, []) if r]
        if not reasons:
            return "Uso il tool più adatto."
        compact = ", ".join(reasons[:3])

        if intent == "hybrid":
            return f"La richiesta combina segnali seller+product ({compact})."
        if intent == "product_search":
            return f"La richiesta ha struttura da ricerca prodotto ({compact})."
        if intent == "seller_analysis":
            return f"La richiesta ha struttura da analisi venditore ({compact})."
        return "La richiesta è conversazionale."

    def _exceeds_tool_budget(self, memory: AgentMemory, tool_name: str) -> bool:
        return memory.tool_call_count(tool_name) >= self.max_calls_per_tool
