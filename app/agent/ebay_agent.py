from __future__ import annotations

import logging
from typing import Dict, Generator, Optional

from sqlalchemy.orm import Session

from app.agent.executor import ToolExecutor
from app.agent.memory import AgentMemory
from app.agent.planner import ReactPlanner
from app.agent.prompts import build_final_answer_prompt
from app.agent.schemas import AgentResponse, AgentRequest, ThinkingEvent, ToolStartEvent, ToolResultEvent, StartEvent, \
    AgentStep, FinalEvent
from app.agent.task_decomposer import decompose_query
from app.agent.tool_registry import ToolContext
from app.services.parser import call_gemini, call_ollama

logger = logging.getLogger(__name__)


class EbayReactAgent:
    def __init__(
        self,
        db: Session,
        user: Optional[object] = None,
    ):
        self.db = db
        self.user = user

    def run(self, request: AgentRequest) -> AgentResponse:
        final_payload: Optional[Dict] = None

        for event in self.run_stream(request):
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

    def run_stream(self, request: AgentRequest) -> Generator[Dict, None, None]:
        max_steps = min(max(int(request.max_steps), 1), 6)

        memory = AgentMemory(user_query=request.query)
        tasks = decompose_query(request.query, request.llm_engine)
        memory.load_tasks(tasks)

        planner = ReactPlanner(llm_engine=request.llm_engine)
        executor = ToolExecutor(
            ToolContext(
                db=self.db,
                user=self.user,
                llm_engine=request.llm_engine,
            )
        )

        trace = []
        final_answer: Optional[str] = None

        yield StartEvent(
            query=request.query,
            llm_engine=request.llm_engine,
            max_steps=max_steps,
            planned_tasks=tasks,
        ).model_dump()

        for step_index in range(1, max_steps + 1):
            decision = planner.decide(
                memory=memory,
                step_index=step_index,
                max_steps=max_steps,
            )

            if decision.intent:
                memory.detected_intent = decision.intent

            if decision.should_stop or decision.action is None:
                final_answer = decision.final_answer or self._finalize(
                    memory=memory,
                    llm_engine=request.llm_engine,
                )

                yield ThinkingEvent(
                    step=step_index,
                    message="Ho raccolto abbastanza informazioni.",
                ).model_dump()
                break

            memory.register_tool_call(decision.action.tool)

            yield ThinkingEvent(
                step=step_index,
                thought=decision.thought,
                action=decision.action.tool,
            ).model_dump()

            yield ToolStartEvent(
                step=step_index,
                tool=decision.action.tool,
                input=decision.action.input,
            ).model_dump()

            observation = executor.execute(decision.action)
            memory.apply_observation(observation)

            yield ToolResultEvent(
                step=step_index,
                tool=decision.action.tool,
                ok=observation.ok,
                status=observation.status,
                quality=observation.quality,
                summary=observation.summary,
            ).model_dump()

            trace.append(
                AgentStep(
                    step=step_index,
                    thought=decision.thought,
                    action=decision.action.tool,
                    action_input=decision.action.input,
                    observation_summary=observation.summary,
                    status=observation.status,
                )
            )

            if not observation.ok:
                logger.warning(
                    "Tool execution failed at step %s: %s",
                    step_index,
                    observation.error,
                )

                if step_index >= max_steps or planner.should_abort_after_error(memory, decision.action.tool):
                    final_answer = self._finalize(memory, request.llm_engine)
                    break

                continue

            if planner.can_stop_early(memory):
                final_answer = self._finalize(memory, request.llm_engine)
                break

        if final_answer is None:
            final_answer = self._finalize(memory, request.llm_engine)

        yield FinalEvent(
            final_answer=final_answer,
            agent_trace=[s.model_dump() for s in trace] if request.return_trace else [],
            final_data=memory.final_data(),
            steps_used=len(trace),
        ).model_dump()

    def _finalize(self, memory: AgentMemory, llm_engine: str) -> str:
        fallback = self._fallback_final_answer(memory)

        if self._is_fallback_good_enough(memory, fallback):
            return fallback

        prompt = build_final_answer_prompt(
            user_query=memory.user_query,
            scratchpad=memory.scratchpad(),
            final_data=memory.final_data(),
        )

        llm_text = self._call_final_llm(prompt, llm_engine)
        if llm_text and llm_text.strip():
            return llm_text.strip()

        return fallback

    @staticmethod
    def _call_final_llm(prompt: str, llm_engine: str) -> Optional[str]:
        try:
            if llm_engine == "gemini":
                return call_gemini(prompt)
            if llm_engine == "ollama":
                return call_ollama(prompt)
            return None
        except Exception as exc:
            logger.warning("Final answer LLM failed: %s", exc)
            return None

    @staticmethod
    def _is_fallback_good_enough(memory: AgentMemory, fallback: str) -> bool:
        if not fallback:
            return False

        if memory.detected_intent == "conversation":
            return True

        if memory.has_terminal_state("search") or memory.has_terminal_state("seller"):
            return True

        if memory.errors:
            return True

        return bool(fallback)

    def _fallback_final_answer(self, memory: AgentMemory) -> str:
        intent = (memory.detected_intent or "").lower()
        search_payload = memory.search_payload
        seller_payload = memory.seller_payload

        if intent == "conversation" and not search_payload and not seller_payload:
            return "Certo, dimmi pure come posso aiutarti."

        if search_payload and seller_payload:
            return self._build_hybrid_answer(memory)

        if search_payload:
            return self._build_search_answer(memory)

        if seller_payload:
            return self._build_seller_answer(memory)

        if memory.errors:
            return (
                "Non sono riuscito a completare correttamente l'analisi. "
                f"Ultimo errore: {memory.errors[-1]}"
            )

        if intent == "seller_analysis":
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
                pass

        text += "."

        if analysis:
            text += f" {analysis}"

        return text.strip()

    def _build_seller_answer(self, memory: AgentMemory) -> str:
        seller_payload = memory.seller_payload or {}
        seller_name = seller_payload.get("seller_name") or memory.last_seller_name or "il venditore"
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
                pass

        if sentiment is not None:
            try:
                details.append(f"sentiment {round(float(sentiment) * 100)}%")
            except Exception:
                pass

        if count is not None:
            details.append(f"{count} feedback analizzati")

        text = f"Ho analizzato {seller_name}"
        if details:
            text += " con " + ", ".join(details)
        text += "."

        return text
