from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.agent.memory import AgentMemory
from app.agent.prompts import build_planner_prompt
from app.agent.schemas import PlannerOutput, ToolCall
from app.agent.tool_registry import extract_explicit_seller
from app.llm.client import call_llm
from app.services.parser import extract_first_json_object

logger = logging.getLogger(__name__)

VALID_INTENTS = {"conversation", "seller_analysis", "product_search", "hybrid", "comparison", "item_details", "shipping", "market_trends", "deals", "wishlist", "contact_seller", "playwright_search"}

EBAY_ID_RE = re.compile(r"\b(?:v1\|)?\d{12,13}(?:\|\d)?\b|\b\d{12,13}\b", re.IGNORECASE)


class ReactPlanner:
    """
    Planner orientato ai capability-tool.

    """

    def __init__(self, llm_engine: str = "gemini", mcp_client: Optional[Any] = None):
        self.llm_engine = (llm_engine or "gemini").strip().lower()
        self.mcp_client = mcp_client
        self._cached_mcp_catalog: Optional[Dict[str, Any]] = None
        self.max_calls_per_tool = 2
        self.intent_threshold = 0.55
        self.margin_threshold = 0.18
        self.hybrid_threshold = 0.62

    async def decide(
            self,
            memory: AgentMemory,
            step_index: int,
            max_steps: int,
            custom_instructions: Optional[str] = None,
            tone: Optional[str] = None
    ) -> PlannerOutput:
        explicit_seller = extract_explicit_seller(memory.user_query)
        if explicit_seller and not memory.last_seller_name:
            memory.last_seller_name = explicit_seller

        if memory.has_pending_tasks():
            decision = await self._decide_from_task_queue(memory)
            if decision:
                logger.info(f"Planned task decision: {decision.action.tool if decision.action else 'finish'}")
                return decision

        logger.info(f"Deciding for query: {memory.user_query}")

        # Inizializza catalogo MCP precocemente per renderlo disponibile a deterministic_decide
        if self.mcp_client is not None and getattr(self.mcp_client, "is_available", False) and self._cached_mcp_catalog is None:
            try:
                self._cached_mcp_catalog = await self.mcp_client.get_tool_schemas_async()
                logger.info("Early MCP catalog fetch successful.")
            except Exception as e:
                logger.warning("Early MCP catalog fetch failed: %s", e)

        # FIX B:
        # fast path conversazionale prima di qualunque possibile chiamata lenta al planner LLM
        conversation_fast_path = self._conversation_fast_path(memory)
        if conversation_fast_path:
            return conversation_fast_path

        # TENTATIVO PRIMARIO: MCP + LLM Native Tool Calling
        # Sfruttiamo i modelli capaci come Minimax/Qwen etc per parseggiare direttamente il raw user_query in JSON MCP
        # Pre-compute a clean browser search query once per session for playwright mode.
        # Uses the parser service (LLM-powered, Redis-cached) so no hardcoded stop-words.
        if memory.mcp_mode == "playwright_browser" and not getattr(memory, "_browser_query", None):
            await self._precompute_browser_query(memory)

        llm_decision = await self._llm_decide(memory, step_index, max_steps, custom_instructions=custom_instructions, tone=tone)
        if llm_decision:
            return llm_decision

        return await self._safe_fallback_decide(memory)

    def _conversation_fast_path(self, memory: AgentMemory) -> Optional[PlannerOutput]:
        text = (memory.user_query or "").strip().lower()

        if text in {"ciao", "hey", "hello", "salve"}:
            return PlannerOutput(
                thought="Saluto semplice.",
                should_stop=True,
                intent="conversation",
            )
        return None

    async def _precompute_browser_query(self, memory: AgentMemory) -> None:
        """Compute a clean product search query for browser_type using the LLM parser.

        Priority:
        1. Reuse the `semantic_query` from a profile_query observation already in memory.
        2. Call parse_query_service (LLM-powered, Redis-cached) to extract it.
        3. Fall back to the raw user_query unchanged.

        Result is stored as memory._browser_query (transient, not persisted).
        """
        # 1. Reuse profile_query result if already computed this session
        pq_data = memory.tool_states.get("profile_query", {}).get("data") or {}
        semantic = (pq_data.get("parsed") or {}).get("semantic_query", "")
        if semantic:
            memory._browser_query = semantic
            return

        # 2. Call the parser (LLM + Redis cache)
        try:
            from app.services.parser import parse_query_service
            result = await parse_query_service(memory.user_query or "", use_llm=True)
            parsed_semantic = result.get("semantic_query", "")
            memory._browser_query = parsed_semantic if parsed_semantic else (memory.user_query or "")
        except Exception:
            logger.warning("_precompute_browser_query failed, using raw query", exc_info=True)
            memory._browser_query = memory.user_query or ""

    async def _generate_seller_message(self, memory: AgentMemory) -> str:
        """Generate a short, professional message to send to an eBay seller.

        Extracts the user's actual intent from their query and formats it as
        a polite seller message. Falls back to a generic inquiry if LLM fails.
        """
        _fallback = "Salve, volevo richiedere informazioni su questo articolo. Grazie."
        user_query = (memory.user_query or "").strip()
        if not user_query:
            return _fallback

        prompt = (
            "L'utente vuole inviare un messaggio a un venditore eBay. "
            f'La sua richiesta è: "{user_query}"\n\n'
            "Scrivi SOLO il testo del messaggio da inviare al venditore: breve, educato, professionale. "
            "Non includere saluti ridondanti. Non includere spiegazioni. Solo il testo del messaggio. "
            "Massimo 3 frasi."
        )
        message = await self._call_llm(prompt)
        return message.strip() if message else _fallback

    def can_stop_early(self, memory: AgentMemory) -> bool:
        if memory.has_pending_tasks():
            return False

        intent = (memory.detected_intent or "").lower()
        if intent == "conversation":
            return True

        mcp_mode = getattr(memory, "mcp_mode", "standard")
        if mcp_mode == "playwright_browser":
            # In modalità step-by-step, non forziamo mai la terminazione prematura dal backend.
            # Aspettiamo ESCLUSIVAMENTE che l'LLM restituisca l'azione `finish`.
            return False

        return self._intent_is_satisfied(memory, intent)

    def should_abort_after_error(self, memory: AgentMemory, failed_tool: str) -> bool:
        return memory.tool_call_count(failed_tool) >= self.max_calls_per_tool

    async def _decide_from_task_queue(self, memory: AgentMemory) -> Optional[PlannerOutput]:
        task = memory.peek_task()
        if not task:
            return None

        tool = str(task.get("tool") or "").strip()
        if self._cached_mcp_catalog is not None and tool not in self._cached_mcp_catalog:
            logger.warning("Skipping queued task for tool=%s because it is not in MCP catalog.", tool)
            memory.pop_task()
            return None

        if self._exceeds_tool_budget(memory, tool):
            logger.warning("Skipping queued task for tool=%s because budget is exhausted.", tool)
            memory.pop_task()
            return None

        action_input = await self._normalize_action_input(tool, task.get("input") or {}, memory)
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
            custom_instructions: Optional[str] = None,
            tone: Optional[str] = None
    ) -> Optional[PlannerOutput]:
        if self.llm_engine == "rule_based":
            return None

        # DYMANIC MCP CATALOG FETCH
        tool_catalog = {}
        if self.mcp_client and self.mcp_client.is_available:
            try:
                tool_catalog = await self.mcp_client.get_tool_schemas_async()
                self._cached_mcp_catalog = tool_catalog
            except Exception as e:
                logger.error("CRITICAL: Failed to fetch dynamic MCP catalog: %s", e)
                self._cached_mcp_catalog = None
        
        if not tool_catalog:
            logger.warning("Agent operating with empty or missing tool catalog.")

        prompt = build_planner_prompt(
            user_query=memory.user_query,
            scratchpad=memory.scratchpad(),
            step_index=step_index,
            max_steps=max_steps,
            tool_catalog=tool_catalog,
            custom_instructions=custom_instructions,
            tone=tone,
            mcp_mode=getattr(memory, "mcp_mode", "standard"),
        )

        raw = await self._call_llm(prompt)
        if not raw:
            logger.info("Planner LLM returned empty output.")
            return None

        payload = None
        try:
            match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                json_str = extract_first_json_object(raw)

            if json_str:
                data = json.loads(json_str)
                action = data.get("action") or data.get("tool")
                action_input = data.get("action_input") or data.get("parameters")

                if action and action not in {"finish", "stop"}:
                    valid = False
                    if self._cached_mcp_catalog and action in self._cached_mcp_catalog:
                        valid = True

                    if not valid and self._cached_mcp_catalog:
                        logger.warning("LLM hallucinated invalid tool: %s. Falling back.", action)
                        return None

                if action or action_input:
                    logger.info("Planner LLM returned valid JSON action=%s", action)
                    payload = data

        except Exception as exc:
            logger.warning("Failed to parse Planner LLM response as JSON: %s\nResponse: %s", exc, raw[:500])
            return None

        if not payload:
            logger.warning("Planner LLM returned no usable JSON. Raw head=%r", raw[:180])
            return None

        thought = str(payload.get("thought") or "").strip()
        intent = str(payload.get("intent") or "").strip().lower()
        action = str(payload.get("action") or payload.get("tool") or "").strip().lower()
        action_input = payload.get("action_input") or payload.get("parameters") or {}

        if intent not in VALID_INTENTS:
            intent = self._infer_intent(memory)

        if action in {"finish", "stop"}:
            if memory.has_pending_tasks():
                return await self._safe_fallback_decide(memory, forced_intent=intent)

            if not self._intent_is_satisfied(memory, intent):
                return await self._safe_fallback_decide(memory, forced_intent=intent)

            return PlannerOutput(
                thought=thought or "Ho raccolto abbastanza informazioni.",
                should_stop=True,
                intent=intent,
            )

        # Redundant check removed as it's now handled above during parsing

        if self._exceeds_tool_budget(memory, action):
            logger.warning("Planner selected tool=%s but budget is exhausted.", action)
            return await self._safe_fallback_decide(memory, forced_intent=intent)

        normalized_input = await self._normalize_action_input(action, action_input, memory)
        if normalized_input is None:
            logger.warning("Planner selected tool=%s with unusable input=%r.", action, action_input)
            return None

        return PlannerOutput(
            thought=thought or "Procedo con il prossimo step.",
            action=ToolCall(tool=action, input=normalized_input),
            intent=intent,
        )

    async def _safe_fallback_decide(
            self,
            memory: AgentMemory,
            forced_intent: Optional[str] = None,
    ) -> PlannerOutput:
        intent = (forced_intent or memory.detected_intent or self._infer_intent(memory)).lower()

        if memory.has_pending_tasks():
            decision = await self._decide_from_task_queue(memory)
            if decision:
                decision.intent = intent if intent in VALID_INTENTS else decision.intent
                return decision

        if intent == "conversation":
            return PlannerOutput(
                thought="La richiesta è conversazionale.",
                should_stop=True,
                intent="conversation",
            )

        mcp_mode = getattr(memory, "mcp_mode", "standard")
        for tool_name in self._ordered_tools_for_intent(intent, memory):
            # Skip tools not available in the current MCP world
            if self._cached_mcp_catalog and tool_name not in self._cached_mcp_catalog:
                continue
            # In playwright mode, skip contact_seller (standard-only tool) even if catalog unavailable
            if mcp_mode == "playwright_browser" and tool_name == "contact_seller":
                continue
            if self._tool_state_is_terminal(memory, tool_name):
                continue
            if self._exceeds_tool_budget(memory, tool_name):
                continue

            normalized_input = await self._normalize_action_input(tool_name, {}, memory)
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

        return PlannerOutput(
            thought="Termino, non ho altri tool utilizzabili o non ho i requisiti per procedere.",
            should_stop=True,
            intent=intent if intent in VALID_INTENTS else self._infer_intent(memory),
            final_answer="Non riesco a procedere oltre con i dati forniti." if intent not in ("conversation", "product_search") else None
        )

    async def _call_llm(self, prompt: str) -> Optional[str]:
        try:
            result, _ = await call_llm(prompt)
            return result
        except Exception as exc:
            logger.warning("Planner LLM failed: %s", exc)
        return None

    async def _normalize_action_input(
            self,
            action: str,
            action_input: Dict[str, Any],
            memory: AgentMemory,
    ) -> Optional[Dict[str, Any]]:
        # Without local registries, we rely entirely on the MCP server (and Pydantic logic inside tools) for validation.
        # However, the deterministic path (and safe fallback) calls this method with an empty action_input={}.
        # We must therefore infer the required arguments from the user query here.
        
        q = memory.user_query or ""
        text = q.strip()
        lowered = text.lower()
        
        if action == "search_products" and not action_input.get("query"):
            # A basic normalizer for search as fallback
            from app.mcp.normalizers import clean_search_query
            action_input["query"] = clean_search_query(text) or text

        elif action == "browser_navigate" and not action_input.get("url"):
            action_input["url"] = "https://www.ebay.it"

        elif action == "browser_type" and not action_input.get("text"):
            # Use the LLM-parsed semantic query (pre-computed in decide()) when available.
            # Falls back to raw user_query if parsing was not performed.
            search_text = getattr(memory, "_browser_query", None) or text
            action_input["selector"] = "#gh-ac"
            action_input["text"] = search_text
            action_input["press_enter"] = True

        elif action in {"analyze_seller", "contact_seller"} and not action_input.get("seller_name"):
            from app.agent.tool_registry import extract_explicit_seller
            seller = extract_explicit_seller(text) or getattr(memory, "last_seller_name", None)
            if not seller:
                return None  # seller_name è obbligatorio per entrambi i tool
            action_input["seller_name"] = seller

        elif action == "compare_products" and not action_input.get("queries"):
            action_input["queries"] = text

        elif action == "market_trends" and not action_input.get("query"):
            action_input["query"] = text

        elif action in {"get_item_details", "get_shipping_costs"} and not action_input.get("item_id"):
            id_match = EBAY_ID_RE.search(text)
            if id_match:
                action_input["item_id"] = id_match.group(0)
                logger.info(f"Auto-extracted item_id for {action}: {action_input['item_id']}")
            else:
                return None # Indispensabile per questi tool

        elif action == "get_ebay_deals":
            # 1. Estrarre Category ID più robusto
            if not action_input.get("category_id"):
                # Cerca patterns come "ID: 9355", "(9355)", "categoria 9355"
                cat_match = re.search(r"(?:id|cat|categoria)[:\s]*(\d{4,8})", lowered)
                if not cat_match:
                    # Fallback: cerca un numero di 4-8 cifre tra parentesi o a fine stringa
                    cat_match = re.search(r"\((\d{4,8})\)|(?:\s|^)(\d{4,8})(?:\s|$)", lowered)
                
                if cat_match:
                    # Prendi il primo gruppo catturato non nullo
                    cat_id = next((g for g in cat_match.groups() if g), None)
                    if cat_id:
                        action_input["category_id"] = cat_id
                        logger.info("Auto-extracted category_id for get_ebay_deals: %s", cat_id)

            # 2. Estrarre Query come fallback (se non già presente)
            if not action_input.get("query"):
                # Rimuoviamo il rumore (ID...) e le parole chiave del tool
                q_clean = re.sub(r"\([^\)]*id[:\s]*\d+[^\)]*\)", "", lowered, flags=re.IGNORECASE)
                q_clean = re.sub(r"\b(cerca|offerte|deals|categoria|id|per|la|del|giorno|migliori|🏷️)\b", "", q_clean, flags=re.IGNORECASE)
                q_clean = re.sub(r"\s+", " ", q_clean).strip()
                if q_clean:
                    action_input["query"] = q_clean
                    logger.info("Auto-extracted query for get_ebay_deals: %s", q_clean)
        
        elif action == "manage_wishlist":
            # Se l'azione è 'add' e manca ebay_id, proviamo a prenderlo dai top_results del scratchpad
            if action_input.get("action") == "add" and not action_input.get("ebay_id"):
                scratchpad = memory.scratchpad()
                results = []
                if isinstance(scratchpad, dict):
                    results = scratchpad.get("results", [])
                elif isinstance(scratchpad, list):
                    # Cerca l'ultimo risultato di ricerca nel scratchpad
                    for item in reversed(scratchpad):
                        if item.get("tool") == "search_products":
                            results = item.get("results", [])
                            break
                
                if results and len(results) > 0:
                    top = results[0]
                    action_input["ebay_id"] = top.get("ebay_id")
                    if not action_input.get("title"):
                        action_input["title"] = top.get("title")
                    if not action_input.get("price"):
                        action_input["price"] = top.get("price")
                    if not action_input.get("currency"):
                        action_input["currency"] = top.get("currency")
                    if not action_input.get("image_url"):
                        action_input["image_url"] = top.get("image_url")
                    if not action_input.get("url"):
                        action_input["url"] = top.get("url")
                    if not action_input.get("seller_name"):
                        action_input["seller_name"] = top.get("seller_name")
                    logger.info("Auto-extracted product details for manage_wishlist 'add' from scratchpad: %s", action_input["ebay_id"])

        elif action == "ebay_scrape":
            # Estraiamo la query pulendo i tag UI se presenti
            raw_q = action_input.get("query") or memory.user_query
            # Mostra il browser solo se esplicitamente richiesto nel messaggio o nell'input
            visible_requested = (
                "modalità visibile" in (raw_q or "").lower()
                or action_input.get("visible", False)
            )
            clean_q = re.sub(r"Cerca su eBay con Playwright \(MODALITÀ VISIBILE\):", "", raw_q, flags=re.IGNORECASE).strip()
            clean_q = clean_q.replace("🌐", "").strip()

            return {
                "query": clean_q,
                "visible": visible_requested,
            }

        elif action == "contact_seller_playwright":
            if not action_input.get("product_url"):
                from app.agent.tool_registry import extract_explicit_seller
                seller = extract_explicit_seller(text) or getattr(memory, "last_seller_name", None)

                if seller:
                    action_input["product_url"] = (
                        f"https://www.ebay.it/cnt/IntermediatedFAQ?seller_name={seller}"
                    )
                    logger.info("contact_seller_playwright: using IntermediatedFAQ URL for seller=%s", seller)
                else:
                    # In playwright mode: read the current browser page URL from tool_states
                    # (BrowserManager stores last navigation result there)
                    browser_url = None
                    from urllib.parse import urlparse as _urlparse
                    for browser_tool in ("browser_click", "browser_get_view", "browser_type", "browser_navigate"):
                        state_data = (memory.tool_states.get(browser_tool) or {}).get("data") or {}
                        url = state_data.get("url", "")
                        parsed_netloc = _urlparse(url).netloc.lower() if url else ""
                        if parsed_netloc and ("ebay." in parsed_netloc):
                            browser_url = url
                            break

                    if browser_url:
                        action_input["product_url"] = browser_url
                        logger.info("contact_seller_playwright: using current browser URL=%s", browser_url)
                    else:
                        # Last resort: standard mode search payload
                        scratchpad = memory.scratchpad()
                        top_results = scratchpad.get("top_results") or []
                        if top_results and top_results[0].get("url"):
                            action_input["product_url"] = top_results[0]["url"]
                        else:
                            return None  # No URL and no seller name — cannot proceed

            if not action_input.get("message"):
                cached_msg = getattr(memory, "_seller_message", None)
                if cached_msg is None:
                    cached_msg = await self._generate_seller_message(memory)
                    memory._seller_message = cached_msg
                action_input["message"] = cached_msg

        return action_input

    def _intent_is_satisfied(self, memory: AgentMemory, intent: str) -> bool:
        if intent == "conversation":
            return True

        mcp_mode = getattr(memory, "mcp_mode", "standard")
        if mcp_mode == "playwright_browser":
            # In modalità visiva iterativa, ci fidiamo ciecamente della decisione di finish del LLM
            # basandoci solo sul fatto che abbia effettivamente interagito almeno una volta
            return len(memory.tool_states) > 0

        tools = self._ordered_tools_for_intent(intent, memory)
        if not tools:
            return memory.has_any_terminal_state()

        if intent in {"hybrid", "shipping", "item_details", "wishlist"}:
            return all(self._tool_state_is_terminal(memory, tool_name) for tool_name in tools)

        return any(self._tool_state_is_terminal(memory, tool_name) for tool_name in tools)

    def _ordered_tools_for_intent(self, intent: str, memory: Optional[AgentMemory] = None) -> list[str]:
        # In Playwright mode we now have full parity of tools.
        # We can use the standard mapping.

        seller_tool = "analyze_seller"
        search_tool = "search_products"
        compare_tool = "compare_products"

        explicit_id = False
        if memory and memory.user_query:
            if EBAY_ID_RE.search(memory.user_query):
                explicit_id = True

        if intent == "comparison":
            return [compare_tool]
        if intent == "seller_analysis":
            return [seller_tool]
        if intent == "product_search":
            mcp_mode = getattr(memory, "mcp_mode", "standard") if memory else "standard"
            if mcp_mode == "playwright_browser":
                return ["browser_navigate", "browser_type"]
            return [search_tool]
        if intent == "playwright_search":
            return ["browser_navigate", "browser_type"]
        if intent == "item_details":
            return ["get_item_details"] if explicit_id else [search_tool, "get_item_details"]
        if intent == "shipping":
            return ["get_shipping_costs"] if explicit_id else [search_tool, "get_shipping_costs"]
        if intent == "hybrid":
            ordered = [search_tool, seller_tool, "get_item_details", "get_shipping_costs"]
            return ordered
        if intent == "market_trends":
            return ["market_trends"]
        if intent == "deals":
            return ["get_ebay_deals"]
        if intent == "wishlist":
            return [search_tool, "manage_wishlist"] if not explicit_id else ["manage_wishlist"]
        if intent == "contact_seller":
            mcp_mode = getattr(memory, "mcp_mode", "standard") if memory else "standard"
            if mcp_mode == "playwright_browser" or (memory and "playwright" in (memory.user_query or "").lower()):
                return ["contact_seller_playwright", "contact_seller"]
            return ["contact_seller"]
        return []

    def _tool_state_is_terminal(self, memory: AgentMemory, tool_name: str) -> bool:
        return memory.has_terminal_state(tool_name)

    def _tool_matches_any_tag(self, tool_name: str, tags: set[str]) -> bool:
        # Retrocompatibilità per infer intent_with_confidence su task pendenti
        if not self._cached_mcp_catalog:
            return False
        schema = self._cached_mcp_catalog.get(tool_name, {})
        desc_lower = str(schema.get("description", "")).lower()
        return any(tag in desc_lower for tag in tags)

    def _infer_intent(self, memory: AgentMemory) -> str:
        q = (memory.user_query or "").lower().strip()
        if not q or q in {"ciao", "hey", "hello", "salve"}:
            return "conversation"
        if any(w in q for w in ["venditore", "affidabilità", "feedback", "recensioni"]):
            return "seller_analysis"
        if any(w in q for w in ["confronta", "compara", "differenza"]):
            return "comparison"
        
        seller = extract_explicit_seller(q)
        if seller:
            # Se parla anche di prodotti (non solo del venditore), è un hybrid
            if len(q.split()) > 2:
                return "hybrid"
            return "seller_analysis"
            
        if any(w in q for w in ["spedizione", "spedirlo", "costa spedire"]):
            return "shipping"
        if any(w in q for w in ["contatta", "scrivi", "messaggio"]):
            return "contact_seller"
        if ("andamento" in q and "prezz" in q) or "market" in q or "statistiche" in q:
            return "market_trends"
        return "product_search"

    def _exceeds_tool_budget(self, memory: AgentMemory, tool_name: str) -> bool:
        budget = self.max_calls_per_tool
        if tool_name.startswith("browser_"):
            if tool_name == "browser_navigate":
                budget = 1  # Navigate to eBay homepage only once
            elif tool_name == "browser_type":
                budget = 1  # Search once; LLM can call it explicitly again if needed
            else:
                budget = 10 # get_view, click etc.
        return memory.tool_call_count(tool_name) >= budget

