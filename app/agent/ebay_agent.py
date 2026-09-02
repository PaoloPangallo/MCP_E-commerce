from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

from sqlalchemy.orm import Session

from app.agent.executor import ToolExecutor
from app.agent.memory import AgentMemory, MemoryService
from app.agent.planner import ReactPlanner
from app.agent.prompts import build_final_answer_prompt, CONVERSATION_ANSWER_SYSTEM_PROMPT
from app.agent.schemas import (
    AgentRequest,
    AgentResponse,
    AgentStep,
    AnswerChunkEvent,
    FinalEvent,
    StartEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
    VisionAnalysisEvent,
)
from app.agent.task_decomposer import decompose_query
from app.agent.tool_registry import analyze_user_query
from app.mcp.client import MCPToolClient
from app.llm.client import call_ollama_cloud, call_llm_stream

logger = logging.getLogger(__name__)

# Quanto della descrizione vision usare come query quando il modello non
# restituisce né brand né tag utilizzabili.
MAX_VISION_QUERY_CHARS = 120

_MCP_DEFAULT_URLS: dict[str, str] = {
    "standard": os.getenv("MCP_SERVER_URL") or "http://127.0.0.1:8050/mcp/standard/mcp",
    "playwright_browser": os.getenv("MCP_PLAYWRIGHT_URL") or "http://127.0.0.1:8050/mcp/playwright/mcp",
}


class EbayReactAgent:
    def __init__(
        self,
        db: Session,
        user: Optional[object] = None,
        mcp_server_url: Optional[str] = None,
        strict_mcp: Optional[bool] = None,
        prefer_mcp: bool = True,
        mcp_mode: str = "standard",
    ) -> None:
        self.db = db
        self.user = user
        self.memory_service = MemoryService()
        self.prefer_mcp = bool(prefer_mcp)
        self.mcp_mode = mcp_mode

        # Resolve MCP server URL from mcp_mode if not explicitly provided
        self.mcp_server_url = (
            mcp_server_url
            or _MCP_DEFAULT_URLS.get(mcp_mode)
            or _MCP_DEFAULT_URLS["standard"]
        )

        if strict_mcp is None:
            strict_mcp = os.getenv("STRICT_MCP", "false").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }

        self.strict_mcp = bool(strict_mcp)

        # Sempre creare un client per-request: viene connesso tramite
        # `async with self.mcp_client` all'inizio di run_stream, garantendo
        # che _session sia sempre inizializzata prima dell'esecuzione dei tool.
        self.mcp_client = MCPToolClient(
            server_url=self.mcp_server_url,
            enabled=self.prefer_mcp,
        )

        logger.info(
            "EbayReactAgent initialized | prefer_mcp=%s | strict_mcp=%s | mcp_server_url=%s | mcp_mode=%s",
            self.prefer_mcp,
            self.strict_mcp,
            self.mcp_server_url,
            self.mcp_mode,
        )

    async def run(self, request: AgentRequest) -> AgentResponse:
        final_payload: Optional[Dict[str, Any]] = None

        async for event in self.run_stream(request):
            if event.get("type") == "final":
                final_payload = event

        if not final_payload:
            return AgentResponse(
                user_query=request.query,
                final_answer="Non sono riuscito a completare l'analisi.",
                agent_trace=[],
                final_data={},
                steps_used=0,
            )

        return AgentResponse(
            user_query=request.query,
            final_answer=final_payload.get("final_answer", "Analisi non disponibile.") if final_payload else "Analisi non disponibile.",
            agent_trace=(final_payload.get("agent_trace", []) if final_payload else []) if request.return_trace else [],
            final_data=final_payload.get("final_data", {}) if final_payload else {},
            steps_used=final_payload.get("steps_used", 0) if final_payload else 0,
        )

    async def run_stream(self, request: AgentRequest) -> AsyncGenerator[Dict[str, Any], None]:
        max_steps = min(max(int(request.max_steps or 4), 1), 6)

        logger.info(
            "Agent run_stream started | query=%s | llm_engine=%s | max_steps=%s | prefer_mcp=%s | strict_mcp=%s",
            request.query,
            request.llm_engine,
            max_steps,
            self.prefer_mcp,
            self.strict_mcp,
        )

        # 1. Vision Pre-processing (IMMEDIATELY, to enrich query before planning)
        vision_desc_temp = None
        if getattr(request, "image", None):
            from app.llm.vision import describe_image_with_vision
            yield ThinkingEvent(
                step=0,
                message="Sto analizzando la tua immagine con Qwen-VL...",
            ).model_dump()

            vision_data = await describe_image_with_vision(request.image)

            if not vision_data:
                # Fallire in silenzio qui significa proseguire con query vuota:
                # il planner ripiegherebbe sull'intent "conversation" e l'utente
                # riceverebbe la risposta del turno PRECEDENTE. Meglio fermarsi.
                logger.error(
                    "AGENT: Vision analysis failed — interrompo il turno (query=%r)",
                    request.query,
                )
                self.memory_service.clear_vision_description(self.user)
                yield {
                    "type": "error",
                    "message": (
                        "Non sono riuscito ad analizzare l'immagine. "
                        "Riprova, magari con una foto più leggera o più nitida."
                    ),
                }
                return

            desc_text = (vision_data.get("description") or "").strip()
            tags = vision_data.get("tags") or []
            brand = vision_data.get("brand")
            condition = vision_data.get("condition_clues")
            try:
                confidence = float(vision_data.get("confidence", 0.95))
            except (TypeError, ValueError):
                confidence = 0.95

            search_additive = []
            if brand:
                search_additive.append(brand)
            if tags:
                search_additive.extend(tags[:4])  # Max 4 tag per compattezza

            compact_str = " ".join(search_additive).strip()
            if not compact_str:
                # Nessun brand e nessun tag (es. fallback su JSON non parsabile):
                # senza questo la query si ridurrebbe al solo prefisso "Trova:".
                compact_str = desc_text[:MAX_VISION_QUERY_CHARS].strip()

            original_query = request.query.strip()

            if not compact_str and not original_query:
                # Il modello non ha prodotto nulla di utilizzabile e l'utente non ha
                # scritto niente: proseguire significherebbe mandare al planner una
                # query vuota, che ripiega su "conversation" e risponde sullo storico.
                logger.error("AGENT: Vision produced no usable query — interrompo il turno")
                self.memory_service.clear_vision_description(self.user)
                yield {
                    "type": "error",
                    "message": (
                        "Ho analizzato l'immagine ma non sono riuscito a ricavarne "
                        "abbastanza dettagli per una ricerca. Prova con un'altra foto."
                    ),
                }
                return

            if not original_query:
                request.query = f"Trova: {compact_str}".strip()
            elif compact_str:
                request.query = f"{original_query} ({compact_str})".strip()
            else:
                request.query = original_query

            vision_desc_temp = desc_text
            logger.info("AGENT: Query enriched with vision data (COMPACT): %s", request.query)

            # EMETTIAMO EVENTO VISIONE PER LA CARD DEDICATA
            yield VisionAnalysisEvent(
                description=desc_text,
                tags=tags,
                brand=brand,
                condition_clues=condition,
                confidence=confidence
            ).model_dump()

        # 2. Initialization
        memory = self.memory_service.hydrate_request_state(
            user_query=request.query,
            user=self.user,
        )
        if getattr(request, "image", None):
            # Turno con immagine: la descrizione idratata appartiene alla foto
            # precedente e non deve sopravvivere a quella nuova.
            memory.vision_description = vision_desc_temp
        memory.mcp_mode = self.mcp_mode

        try:
            tasks = decompose_query(request.query, request.llm_engine)
        except Exception as exc:
            logger.warning("Task decomposition failed: %s", exc)
            tasks = []

        try:
            memory.load_tasks(tasks)
        except Exception as exc:
            logger.warning("Unable to load tasks into memory: %s", exc)

        planner = ReactPlanner(
            llm_engine=request.llm_engine,
            mcp_client=self.mcp_client,
        )

        executor = ToolExecutor(
            context=type("Context", (), {
                "db": self.db,
                "user": self.user,
                "llm_engine": request.llm_engine,
                "user_query": memory.user_query,
            })(),
            mcp_client=self.mcp_client,
            prefer_mcp=self.prefer_mcp,
            fallback_to_local=not self.strict_mcp,
            mcp_mode=self.mcp_mode,
        )

        trace: List[AgentStep] = []
        final_answer: str = ""
        executed_actions = 0

        yield StartEvent(
            query=request.query,
            llm_engine=request.llm_engine,
            max_steps=max_steps,
            planned_tasks=tasks,
        ).model_dump()

        async with self.mcp_client:
            if self.mcp_client.is_available:
                yield ThinkingEvent(
                    step=0,
                    message="🔄 Sincronizzazione Protocollo MCP in corso..."
                ).model_dump()

            # ---> MCP RESOURCE FETCH: PROFILO UTENTE DINAMICO <---
            user_id = getattr(self.user, "id", None)
            if self.mcp_client.is_available and user_id:
                profile_uri = f"profile://{user_id}"
                raw_profile = await self.mcp_client.read_resource_async(profile_uri)
                if raw_profile:
                    try:
                        import json
                        prof_data = json.loads(raw_profile).get("profile", {})
                        if prof_data:
                            addons = []
                            if prof_data.get("language"):
                                addons.append(f"Rispondi sempre all'utente nella lingua '{prof_data['language']}'")
                            if prof_data.get("favorite_brands"):
                                addons.append(f"L'utente preferisce questi brand: {prof_data['favorite_brands']}")
                            if prof_data.get("price_preference"):
                                addons.append(f"Note sul budget dell'utente: {prof_data['price_preference']}")
                            
                            if addons:
                                extra = " ".join(addons)
                                current = getattr(self.user, "custom_instructions", "") or ""
                                # Inject dynamic instructions into user object
                                setattr(self.user, "custom_instructions", current + " | [MCP PROFILE] " + extra if current else "[MCP PROFILE] " + extra)
                                logger.info("Injected MCP Profile instructions: %s", extra)
                                
                                yield ThinkingEvent(
                                    step=0,
                                    message=f"📥 Profilo Utente MCP acquisito (Brand, Budget, Lingua in memory)."
                                ).model_dump()
                    except Exception as exc:
                        logger.warning("Failed to parse MCP profile resource: %s", exc)

            # ---> MCP PROMPT FETCH: SHOPPING EXPERT <---
            if self.mcp_client.is_available:
                expert_prompt = await self.mcp_client.get_prompt_async("shopping_expert_prompt")
                if expert_prompt:
                    # Se non c'è user in sessione, creiamo un mock object per ospitare le info
                    if not self.user:
                        class MockUser: pass
                        self.user = MockUser()
                    
                    current = getattr(self.user, "custom_instructions", "") or ""
                    # Iniettiamo l'identità MCP in cima alle allocazioni
                    setattr(self.user, "custom_instructions", expert_prompt + "\n\n" + current if current else expert_prompt)
                    logger.info("Injected MCP Identity Prompt.")
                    
                    yield ThinkingEvent(
                        step=0,
                        message="🧠 Identità MCP (shopping_expert_prompt) attivata con successo."
                    ).model_dump()

            for step_index in range(1, max_steps + 1):
                try:
                    decision = await planner.decide(
                        memory=memory,
                        step_index=step_index,
                        max_steps=max_steps,
                        custom_instructions=getattr(self.user, "custom_instructions", None),
                        tone=getattr(self.user, "conversation_tone", None)
                    )
                except Exception as exc:
                    logger.exception("Planner failed at step %s: %s", step_index, exc)
                    memory.errors.append(f"Planner error: {exc}")
                    final_answer = await self._finalize(memory=memory, llm_engine=request.llm_engine)
                    break

                if getattr(decision, "intent", None):
                    memory.detected_intent = decision.intent

                if decision.should_stop or (decision.action is None and not decision.actions):
                    if decision.final_answer:
                        final_answer = decision.final_answer
                    else:
                        yield ThinkingEvent(
                            step=step_index,
                            message="Sto preparando la sintesi finale per te...",
                        ).model_dump()
                        # The actual streaming will happen at the end of the loop
                    break

                planned_actions = decision.planned_actions()
                thought_action = planned_actions[0].tool if planned_actions else None

                yield ThinkingEvent(
                    step=step_index,
                    thought=decision.thought,
                    action=thought_action,
                    message=f"Eseguo {thought_action}..." if thought_action else None
                ).model_dump()

                for action in planned_actions:
                    memory.register_tool_call(action.tool)
                    yield ToolStartEvent(
                        step=step_index,
                        tool=action.tool,
                        input=action.input,
                    ).model_dump()

                try:
                    observations = await executor.execute_many(
                        planned_actions,
                        parallel=decision.run_parallel,
                    )
                except Exception as exc:
                    logger.exception("Executor failed at step %s: %s", step_index, exc)
                    memory.errors.append(f"Executor error: {exc}")

                    if self.strict_mcp:
                        final_answer = (
                            "L'esecuzione tramite MCP è fallita e la modalità strict è attiva."
                        )
                    else:
                        final_answer = await self._finalize(memory=memory, llm_engine=request.llm_engine)
                    break

                for action, observation in zip(planned_actions, observations):
                    memory.apply_observation(observation)
                    executed_actions += 1

                    backend = getattr(observation, "backend", None)
                    if backend:
                        logger.info(
                            "Tool result | step=%s | tool=%s | backend=%s | ok=%s | status=%s",
                            step_index,
                            action.tool,
                            backend,
                            observation.ok,
                            observation.status,
                        )
                    else:
                        logger.info(
                            "Tool result | step=%s | tool=%s | ok=%s | status=%s",
                            step_index,
                            action.tool,
                            observation.ok,
                            observation.status,
                        )
                        # Log data size if present
                        if observation.data and isinstance(observation.data, dict):
                            res_list = observation.data.get("results") or observation.data.get("items")
                            if isinstance(res_list, list):
                                logger.info("Tool data results count: %d", len(res_list))

                    event_payload = ToolResultEvent(
                        step=step_index,
                        tool=action.tool,
                        ok=observation.ok,
                        status=observation.status,
                        quality=observation.quality,
                        summary=observation.summary,
                    ).model_dump()
                    
                    if observation.ok and observation.data:
                        res_count = len(observation.data.get("results", []))
                        logger.info("SSE tool_result payload | results_count=%d | keys=%s",
                                    res_count, list(observation.data.keys()))
                        event_payload["data"] = observation.data

                    if backend:
                        event_payload["backend"] = backend

                    yield event_payload

                    trace.append(
                        AgentStep(
                            step=step_index,
                            thought=decision.thought,
                            action=action.tool,
                            action_input=action.input,
                            observation_summary=observation.summary,
                            status=observation.status,
                        )
                    )

                    if observation.terminal:
                        yield ThinkingEvent(
                            step=step_index,
                            message="Analisi completata. Sto scrivendo la risposta...",
                        ).model_dump()
                        final_answer = await self._finalize(memory=memory, llm_engine=request.llm_engine)
                        break # Break from inner loop, then outer loop will break too

                if final_answer: # If final_answer was set due to observation.terminal
                    break # Break from outer loop

                if any(not obs.ok for obs in observations):
                    failed_tools = [obs.tool for obs in observations if not obs.ok]

                    try:
                        should_abort = (
                            step_index >= max_steps
                            or any(planner.should_abort_after_error(memory, tool) for tool in failed_tools)
                        )
                    except Exception as exc:
                        logger.warning("Planner abort policy failed: %s", exc)
                        should_abort = step_index >= max_steps

                    if should_abort:
                        final_answer = await self._finalize(memory=memory, llm_engine=request.llm_engine)
                        break

                    continue

                try:
                    if planner.can_stop_early(memory):
                        final_answer = await self._finalize(memory=memory, llm_engine=request.llm_engine)
                        break
                except Exception as exc:
                    logger.warning("Planner early-stop check failed: %s", exc)

            # --- FINALIZZAZIONE STREAMING ---
            if not final_answer:
                full_text = ""
                async for chunk in self._finalize_stream(memory=memory, llm_engine=request.llm_engine):
                    if chunk:
                        full_text += chunk
                        yield AnswerChunkEvent(chunk=chunk).model_dump()
                
                final_answer = full_text or self._fallback_final_answer(memory)
            else:
                # Abbiamo già una risposta (es. tool conversation o early stop):
                # emettiamola direttamente senza richiamare l'LLM.
                yield AnswerChunkEvent(chunk=final_answer).model_dump()


            await asyncio.to_thread(self._persist_outcome_safely, memory, final_answer)

            yield FinalEvent(
                final_answer=final_answer,
                agent_trace=[s.model_dump() for s in trace] if request.return_trace else [],
                final_data=memory.final_data(),
                steps_used=executed_actions,
            ).model_dump()

    def _persist_outcome_safely(self, memory: AgentMemory, final_answer: str) -> None:
        try:
            self.memory_service.persist_request_outcome(memory, final_answer)
        except Exception as exc:
            logger.warning("Persisting memory outcome failed: %s", exc)

    async def _finalize_stream(self, memory: AgentMemory, llm_engine: str) -> AsyncGenerator[str, None]:
        intent = (memory.detected_intent or "").lower()

        # PRIORITÀ: Se abbiamo già una risposta generata dal tool conversation, usiamola direttamente.
        # Questo evita doppie chiamate LLM e incoerenze tra trace e risposta finale.
        conv_state = memory.tool_states.get("conversation")
        if conv_state:
            ans = (conv_state.get("data") or {}).get("answer")
            if ans:
                yield ans
                return

        if intent == "conversation":
            # Per l'intento conversazionale, usiamo sempre l'LLM per mantenere il tono da personal shopper.
            prompt = build_final_answer_prompt(
                user_query=memory.user_query,
                scratchpad=memory.scratchpad(),
                final_data=memory.final_data(),
                custom_instructions=getattr(self.user, "custom_instructions", None),
                tone=getattr(self.user, "conversation_tone", None)
            )

            got_any = False
            async for chunk in self._call_final_llm_stream(prompt, llm_engine):
                got_any = True
                yield chunk
            
            if got_any:
                memory.register_llm_call("final")
                return

        # 1. Caso Dettagli/Confronto
        if intent in {"comparison", "item_details"}:
            async for chunk in self._finalize_comparison_stream(memory, llm_engine):
                yield chunk
            return

        # 2. Caso Conversazionale
        if intent == "conversation" and self._should_use_llm_for_final(memory, llm_engine):
            async for chunk in self._finalize_conversation_stream(memory, llm_engine):
                yield chunk
            return

        # 3. Fallback / Intent generico
        fallback = self._fallback_final_answer(memory)
        custom_instructions = getattr(self.user, "custom_instructions", None)

        if custom_instructions and self._should_use_llm_for_final(memory, llm_engine):
            pass  # salta il fallback shortcut → va dritto al LLM sotto
        elif self._is_fallback_good_enough(memory, fallback):
            yield fallback
            return

        if not self._should_use_llm_for_final(memory, llm_engine):
            yield fallback
            return

        # 4. Prompt di default (Personal Shopper)
        async for chunk in self._finalize_default_stream(memory, llm_engine, fallback):
            yield chunk

    async def _finalize_comparison_stream(self, memory: AgentMemory, llm_engine: str) -> AsyncGenerator[str, None]:
        """Gestisce la finalizzazione per intent di confronto o dettagli."""
        if self._should_use_llm_for_final(memory, llm_engine):
            comparison_prompt = build_final_answer_prompt(
                user_query=memory.user_query,
                scratchpad=memory.scratchpad(),
                final_data=memory.final_data(),
                custom_instructions=getattr(self.user, "custom_instructions", None),
                tone=getattr(self.user, "conversation_tone", None)
            )
            got_any = False
            async for chunk in self._call_final_llm_stream(comparison_prompt, llm_engine):
                got_any = True
                yield chunk
            if got_any:
                memory.register_llm_call("final")
                return
        yield self._build_comparison_answer(memory)

    async def _finalize_conversation_stream(self, memory: AgentMemory, llm_engine: str) -> AsyncGenerator[str, None]:
        """Gestisce la finalizzazione per intent puramente conversazionali."""
        conv_history = []
        try:
            conv_history = list(memory.session_memory.conversation_history[-6:])
        except Exception:
            pass

        history_text = ""
        if conv_history:
            lines = []
            for turn in conv_history:
                role = turn.get("role", "")
                content = str(turn.get("content", "")).strip()
                if role == "user":
                    lines.append(f"Utente: {content}")
                elif role == "assistant":
                    lines.append(f"Assistente: {content}")
            history_text = "\n".join(lines)

        conv_prompt = (
            f"{CONVERSATION_ANSWER_SYSTEM_PROMPT}\n\n"
            + (f"--- STORICO CONVERSAZIONE ---\n{history_text}\n--- FINE STORICO ---\n\n" if history_text else "")
            + f"Utente: {memory.user_query}\nAssistente:"
        )

        custom_instructions = getattr(self.user, "custom_instructions", None)
        if custom_instructions:
            conv_prompt = (
                f"{CONVERSATION_ANSWER_SYSTEM_PROMPT}\n"
                f"REGOLA PRIORITÀ ASSOLUTA:\n{custom_instructions}\n\n"
                + (f"--- STORICO ---\n{history_text}\n---\n\n" if history_text else "")
                + f"Utente: {memory.user_query}\nAssistente:"
            )

        got_any = False
        async for chunk in self._call_final_llm_stream(conv_prompt, llm_engine):
            got_any = True
            yield chunk
        if got_any:
            memory.register_llm_call("final")

    async def _finalize_default_stream(self, memory: AgentMemory, llm_engine: str, fallback: str) -> AsyncGenerator[str, None]:
        """Gestisce la finalizzazione standard se nessun altro intent ha consumato lo stream."""
        prompt = build_final_answer_prompt(
            user_query=memory.user_query,
            scratchpad=memory.scratchpad(),
            final_data=memory.final_data(),
            custom_instructions=getattr(self.user, "custom_instructions", None),
            tone=getattr(self.user, "conversation_tone", None)
        )

        got_any = False
        async for chunk in self._call_final_llm_stream(prompt, llm_engine):
            got_any = True
            yield chunk

        if got_any:
            memory.register_llm_call("final")
        else:
            yield fallback

    async def _call_final_llm_stream(self, prompt: str, llm_engine: str) -> AsyncGenerator[str, None]:
        try:
            gen = call_llm_stream(prompt)
            # Aumentato il delay a un minuto come richiesto dall'utente
            try:
                first = await asyncio.wait_for(gen.__anext__(), timeout=60.0)
                yield first
            except asyncio.TimeoutError:
                logger.warning("Agent stream timed out on first chunk (60s)")
                # Invece di tornare una stringa fissa e interrompere, 
                # forniamo un riepilogo testuale basato sui dati reali in memoria.
                yield "Scusa per l'attesa, l'IA è un po' lenta. Ecco una sintesi rapida dei risultati:\n\n"
                
                # Accediamo alla memoria dall'istanza per generare il fallback
                # Nota: qui siamo dentro _call_final_llm_stream che non ha 'memory' come argomento diretto,
                # ma viene chiamata da _finalize_default_stream che ce l'ha.
                # Tuttavia, _fallback_final_answer è disponibile nell'istanza self.
                return 

            # Continua normalmente per il resto
            async for chunk in gen:
                yield chunk

        except Exception as exc:
            logger.warning("Final answer streaming LLM failed: %s", exc)
            yield ""

    async def _finalize(self, memory: AgentMemory, llm_engine: str) -> str:
        """Versione sincrona (non-generator) di finalize per compatibilità."""
        full = ""
        async for chunk in self._finalize_stream(memory, llm_engine):
            full += chunk
        return full

    @staticmethod
    def _is_fallback_good_enough(memory: AgentMemory, fallback: str) -> bool:
        if not fallback:
            return False

        if (memory.detected_intent or "").lower() == "conversation":
            return False

        # Preferiamo SEMPRE l'LLM synthesis per un output Premium da Personal Shopper.
        # Usa il fallback crudo solo in caso di errori critici o engine non adeguato.
        if memory.errors:
            return True

        return False

    def _fallback_final_answer(self, memory: AgentMemory) -> str:
        profile = analyze_user_query(memory.user_query)

        if memory.search_payload and memory.seller_payload:
            return self._build_hybrid_answer(memory)

        if memory.search_payload:
            return self._build_search_answer(memory)

        if memory.seller_payload:
            return self._build_seller_answer(memory)

        if memory.errors:
            return (
                "Non sono riuscito a completare correttamente l'analisi. "
                f"Ultimo errore: {memory.errors[-1]}"
            )

        if (
            profile["conversation_signal"]
            and not profile["seller_signal"]
            and not profile["search_signal"]
        ):
            return "Non riesco a generare una risposta conversazionale in questo momento."

        recent = memory.recent_observations(limit=3)
        if recent:
            joined = " ".join(
                item["summary"].strip()
                for item in recent
                if isinstance(item.get("summary"), str) and item["summary"].strip()
            ).strip()
            if joined:
                return joined

        if profile["seller_signal"]:
            return "Per analizzare il venditore mi serve il suo nome esatto."
            
        # Nota breve sulle istruzioni se presenti nel fallback rule-based
        custom = getattr(self.user, "custom_instructions", None)
        suffix = ""
        if custom:
            suffix = "\n\n(Nota: Ho ricevuto le tue istruzioni personalizzate ma ho riscontrato un problema tecnico nel generare una risposta complessa. Ecco i dati grezzi)."

        return "Non ho raccolto abbastanza informazioni per produrre una risposta utile." + suffix

    def _build_hybrid_answer(self, memory: AgentMemory) -> str:
        seller_part = self._build_seller_answer(memory)
        search_part = self._build_search_answer(memory)
        return f"{seller_part} {search_part}".strip()

    def _build_search_answer(self, memory: AgentMemory) -> str:
        search_payload = memory.search_payload or {}
        results = search_payload.get("results") or []
        analysis = (memory.search_analysis or "").strip()

        if not results:
            return "Ho eseguito la ricerca ma non ho trovato risultati utili per questa richiesta."

        top = results[0]
        title = top.get("title") or "un prodotto"
        price = top.get("price")
        currency = top.get("currency") or "EUR"
        seller_name = (
            top.get("seller_name")
            or top.get("seller_username")
            or memory.last_seller_name
        )
        trust_score = top.get("trust_score")

        text = f"Non sono riuscito a finalizzare l'output per i risultati trovati."
        if search_payload.get("results_count"):
            text = f"Ho trovato {search_payload.get('results_count')} risultati pertinenti per te! I dettagli sono elencati nella tabella tecnica qui in basso."

        return text.strip()

    @staticmethod
    def _should_use_llm_for_final(memory: AgentMemory, llm_engine: str) -> bool:
        """Determina se invocare l'LLM per la sintesi finale."""
        if llm_engine == "rule_based":
            return False

        # Se abbiamo dati strutturati (search, seller, etc), usiamo SEMPRE l'LLM per la sintesi "premium".
        if memory.has_any_terminal_state():
            return True
        
        # Per conversazioni, preferiamo l'LLM (ma evitiamo loop se ha già fallito)
        if memory.llm_call_count("final") >= 1:
            return False

        return True

    def _build_seller_answer(self, memory: AgentMemory) -> str:
        seller_payload = memory.seller_payload or {}
        seller_name = (
            seller_payload.get("seller_name")
            or memory.last_seller_name
            or "il venditore"
        )
        count = seller_payload.get("count")
        trust_score = seller_payload.get("trust_score")
        sentiment = seller_payload.get("sentiment_score")
        error = seller_payload.get("error")

        if error:
            return (
                f"Ho provato ad analizzare il venditore {seller_name}, "
                f"ma non ho trovato feedback consultabili per questo utente."
            )

        details = []
        if trust_score is not None:
            try:
                details.append(f"trust score {round(float(trust_score) * 100)}%")
            except Exception:
                logger.debug("Unable to format seller trust_score=%s", trust_score)
        if sentiment is not None:
            try:
                details.append(f"sentiment {round(float(sentiment) * 100)}%")
            except Exception:
                logger.debug("Unable to format sentiment_score=%s", sentiment)
        if count is not None:
            details.append(f"{count} feedback analizzati")

        text = f"Ho analizzato {seller_name}"
        if details:
            text += " con " + ", ".join(details)
        text += "."
        return text

    def _build_comparison_answer(self, memory: AgentMemory) -> str:
        payload = memory.compare_payload or {}
        winner = payload.get("winner") or {}
        items_count = payload.get("candidates_found", 0)
        winner_reason = payload.get("winner_reason")

        if not winner or items_count == 0:
            return "Ho provato a confrontare i prodotti ma non ho trovato abbastanza dati per un consiglio affidabile."

        title = winner.get("title") or "prodotto"
        price = winner.get("price")
        currency = winner.get("currency") or "EUR"

        text = f"Ho confrontato {items_count} prodotti per te. "
        text += f"Il vincitore consigliato è '{title}'"
        if price is not None:
            text += f", al prezzo di {price} {currency}"
        text += "."

        if winner_reason:
            text += f" {winner_reason}"

        return text.strip()