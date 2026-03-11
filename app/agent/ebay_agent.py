from __future__ import annotations

import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

from sqlalchemy.orm import Session

from app.agent.executor import ToolExecutor
from app.agent.memory import AgentMemory, MemoryService
from app.agent.planner import ReactPlanner
from app.agent.prompts import build_final_answer_prompt
from app.agent.schemas import (
    AgentRequest,
    AgentResponse,
    AgentStep,
    FinalEvent,
    StartEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from app.agent.task_decomposer import decompose_query
from app.agent.tool_registry import ToolContext, analyze_user_query
from app.mcp.client import MCPToolClient
from app.services.parser import call_gemini, call_ollama

logger = logging.getLogger(__name__)


class EbayReactAgent:
    def __init__(
        self,
        db: Session,
        user: Optional[object] = None,
        mcp_server_url: Optional[str] = None,
        strict_mcp: Optional[bool] = None,
        prefer_mcp: bool = True,
        mcp_client: Optional[object] = None,  # inject app-level singleton if available
    ) -> None:
        self.db = db
        self.user = user
        self.memory_service = MemoryService()
        self.prefer_mcp = bool(prefer_mcp)

        self.mcp_server_url = (
            mcp_server_url
            or os.getenv("MCP_SERVER_URL")
            or "http://127.0.0.1:8050/mcp/mcp"
        )

        if strict_mcp is None:
            strict_mcp = os.getenv("STRICT_MCP", "false").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }

        self.strict_mcp = bool(strict_mcp)

        if mcp_client is not None:
            # Use the externally provided (app-level) MCP client
            self.mcp_client = mcp_client
        else:
            # Fallback: create a per-request client (previous behaviour)
            self.mcp_client = MCPToolClient(
                server_url=self.mcp_server_url,
                enabled=self.prefer_mcp,
            )

        logger.info(
            "EbayReactAgent initialized | prefer_mcp=%s | strict_mcp=%s | mcp_server_url=%s",
            self.prefer_mcp,
            self.strict_mcp,
            self.mcp_server_url,
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
            final_answer=final_payload.get("final_answer", "Analisi non disponibile."),
            agent_trace=final_payload.get("agent_trace", []) if request.return_trace else [],
            final_data=final_payload.get("final_data", {}),
            steps_used=final_payload.get("steps_used", 0),
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

        memory = self.memory_service.hydrate_request_state(
            user_query=request.query,
            user=self.user,
        )

        try:
            tasks = decompose_query(request.query, request.llm_engine)
        except Exception as exc:
            logger.warning("Task decomposition failed: %s", exc)
            tasks = []

        try:
            memory.load_tasks(tasks)
        except Exception as exc:
            logger.warning("Unable to load tasks into memory: %s", exc)

        planner = ReactPlanner(llm_engine=request.llm_engine)

        executor = ToolExecutor(
            context=ToolContext(
                db=self.db,
                user=self.user,
                llm_engine=request.llm_engine,
            ),
            mcp_client=self.mcp_client,
            prefer_mcp=self.prefer_mcp,
            fallback_to_local=not self.strict_mcp,
        )

        trace: List[AgentStep] = []
        final_answer: Optional[str] = None
        executed_actions = 0

        yield StartEvent(
            query=request.query,
            llm_engine=request.llm_engine,
            max_steps=max_steps,
            planned_tasks=tasks,
        ).model_dump()

        async with self.mcp_client:
            for step_index in range(1, max_steps + 1):
                try:
                    decision = await planner.decide(
                        memory=memory,
                        step_index=step_index,
                        max_steps=max_steps,
                    )
                except Exception as exc:
                    logger.exception("Planner failed at step %s: %s", step_index, exc)
                    memory.errors.append(f"Planner error: {exc}")
                    final_answer = await self._finalize(memory=memory, llm_engine=request.llm_engine)
                    break

                if getattr(decision, "intent", None):
                    memory.detected_intent = decision.intent

                if decision.should_stop or (decision.action is None and not decision.actions):
                    final_answer = decision.final_answer or await self._finalize(
                        memory=memory,
                        llm_engine=request.llm_engine,
                    )
                    yield ThinkingEvent(
                        step=step_index,
                        message="Sto preparando la risposta.",
                    ).model_dump()
                    break

                planned_actions = decision.planned_actions()
                thought_action = planned_actions[0].tool if planned_actions else None

                yield ThinkingEvent(
                    step=step_index,
                    thought=decision.thought,
                    action=thought_action,
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

                    event_payload = ToolResultEvent(
                        step=step_index,
                        tool=action.tool,
                        ok=observation.ok,
                        status=observation.status,
                        quality=observation.quality,
                        summary=observation.summary,
                    ).model_dump()

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
                        final_answer = observation.summary or await self._finalize(
                            memory=memory,
                            llm_engine=request.llm_engine,
                        )

                        self._persist_outcome_safely(memory, final_answer)

                        yield FinalEvent(
                            final_answer=final_answer,
                            agent_trace=[s.model_dump() for s in trace] if request.return_trace else [],
                            final_data=memory.final_data(),
                            steps_used=executed_actions,
                        ).model_dump()
                        return

                    if not observation.ok:
                        logger.warning(
                            "Tool execution failed | step=%s | tool=%s | error=%s",
                            step_index,
                            action.tool,
                            observation.error,
                        )

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

            if final_answer is None:
                if memory.has_any_terminal_state():
                    last = memory.recent_observations(limit=1)
                    if last:
                        final_answer = last[0].get("summary") or await self._finalize(
                            memory=memory,
                            llm_engine=request.llm_engine,
                        )
                    else:
                        final_answer = await self._finalize(memory=memory, llm_engine=request.llm_engine)
                else:
                    final_answer = await self._finalize(memory=memory, llm_engine=request.llm_engine)

            import asyncio
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

    async def _finalize(self, memory: AgentMemory, llm_engine: str) -> str:
        intent = (memory.detected_intent or "").lower()

        if intent == "conversation":
            prompt = build_final_answer_prompt(
                user_query=memory.user_query,
                scratchpad=memory.scratchpad(),
                final_data=memory.final_data(),
            )

            llm_text = await self._call_final_llm(prompt, llm_engine)
            if llm_text and llm_text.strip():
                memory.register_llm_call("final")
                return llm_text.strip()

            return "Non riesco a generare una risposta conversazionale in questo momento."

        if intent == "comparison":
            return self._build_comparison_answer(memory)

        fallback = self._fallback_final_answer(memory)

        if self._is_fallback_good_enough(memory, fallback):
            return fallback

        if not self._should_use_llm_for_final(memory, llm_engine):
            return fallback

        prompt = build_final_answer_prompt(
            user_query=memory.user_query,
            scratchpad=memory.scratchpad(),
            final_data=memory.final_data(),
        )

        llm_text = await self._call_final_llm(prompt, llm_engine)
        if llm_text and llm_text.strip():
            memory.register_llm_call("final")
            return llm_text.strip()

        return fallback

    @staticmethod
    async def _call_final_llm(prompt: str, llm_engine: str) -> Optional[str]:
        import asyncio
        try:
            if llm_engine == "gemini":
                return await asyncio.to_thread(call_gemini, prompt)
            if llm_engine == "ollama":
                return await asyncio.to_thread(call_ollama, prompt)
            return None
        except Exception as exc:
            logger.warning("Final answer LLM failed: %s", exc)
            return None

    @staticmethod
    def _is_fallback_good_enough(memory: AgentMemory, fallback: str) -> bool:
        if not fallback:
            return False

        if (memory.detected_intent or "").lower() == "conversation":
            return False

        if memory.search_payload or memory.seller_payload:
            return True

        if memory.errors:
            return True

        return False

    @staticmethod
    def _should_use_llm_for_final(memory: AgentMemory, llm_engine: str) -> bool:
        if llm_engine == "rule_based":
            return False

        if memory.llm_call_count("final") >= 1:
            return False

        if (memory.detected_intent or "").lower() == "conversation":
            return True

        useful_states = [
            state
            for state in memory.tool_states.values()
            if state.get("quality") in {"partial", "good"}
        ]
        if len(useful_states) >= 2 and not memory.errors:
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

        return "Non ho raccolto abbastanza informazioni per produrre una risposta utile."

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

        text = f"Il risultato migliore che ho trovato è '{title}'"
        if price is not None:
            text += f", al prezzo di {price} {currency}"
        if seller_name:
            text += f", venduto da {seller_name}"
        if trust_score is not None:
            try:
                text += f" con trust score {round(float(trust_score) * 100)}%"
            except Exception:
                logger.debug("Unable to format trust_score=%s", trust_score)
        text += "."

        if analysis:
            text += f" {analysis}"
        elif search_payload.get("results_count"):
            text += f" Ho trovato in totale {search_payload.get('results_count')} risultati."

        return text.strip()

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