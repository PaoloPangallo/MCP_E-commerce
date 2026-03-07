from __future__ import annotations

import logging
from typing import Dict, Generator, Optional

from sqlalchemy.orm import Session

from app.agent.executor import ToolExecutor
from app.agent.memory import AgentMemory
from app.agent.planner import ReactPlanner
from app.agent.prompts import build_final_answer_prompt
from app.agent.schemas import AgentRequest, AgentResponse, AgentStep
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
        """
        Esecuzione non-streaming ottimizzata:
        non costruisce tutta la lista eventi, ma tiene solo il finale e la trace.
        """
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
        max_steps = min(max(int(request.max_steps), 1), 5)

        memory = AgentMemory(user_query=request.query)
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

        yield {
            "type": "start",
            "query": request.query,
            "llm_engine": request.llm_engine,
            "max_steps": max_steps,
        }

        for step_index in range(1, max_steps + 1):
            decision = planner.decide(
                memory=memory,
                step_index=step_index,
                max_steps=max_steps,
            )

            # --------------------------------------------------
            # STOP
            # --------------------------------------------------

            if decision.should_stop or decision.action is None:
                final_answer = decision.final_answer or self._finalize(
                    memory=memory,
                    llm_engine=request.llm_engine,
                )

                yield {
                    "type": "thinking",
                    "step": step_index,
                    "message": "Ho raccolto abbastanza informazioni.",
                }
                break

            # --------------------------------------------------
            # THINKING
            # --------------------------------------------------

            yield {
                "type": "thinking",
                "step": step_index,
                "thought": decision.thought,
                "action": decision.action.tool,
            }

            # --------------------------------------------------
            # TOOL START
            # --------------------------------------------------

            yield {
                "type": "tool_start",
                "step": step_index,
                "tool": decision.action.tool,
                "input": decision.action.input,
            }

            # --------------------------------------------------
            # TOOL EXECUTION
            # --------------------------------------------------

            observation = executor.execute(decision.action)
            memory.apply_observation(observation)

            # --------------------------------------------------
            # TOOL RESULT
            # --------------------------------------------------

            yield {
                "type": "tool_result",
                "step": step_index,
                "tool": decision.action.tool,
                "ok": observation.ok,
                "summary": observation.summary,
            }

            step = AgentStep(
                step=step_index,
                thought=decision.thought,
                action=decision.action.tool,
                action_input=decision.action.input,
                observation_summary=observation.summary,
                status="ok" if observation.ok else "error",
            )
            trace.append(step)

            if not observation.ok:
                final_answer = self._finalize(memory, request.llm_engine)
                break

            # stop anticipato se search già sufficiente e query non richiede seller
            if memory.search_payload is not None and planner.can_stop_early(memory):
                final_answer = self._finalize(memory, request.llm_engine)
                break

        if final_answer is None:
            final_answer = self._finalize(memory, request.llm_engine)

        yield {
            "type": "final",
            "final_answer": final_answer,
            "agent_trace": [s.model_dump() for s in trace] if request.return_trace else [],
            "final_data": memory.final_data(),
            "steps_used": len(trace),
        }

    def _finalize(self, memory: AgentMemory, llm_engine: str) -> str:
        """
        Finalizzazione più aggressiva:
        - evita LLM finale quando abbiamo già una risposta fallback buona
        - usa LLM solo se serve arricchire davvero
        """
        fallback = self._fallback_final_answer(memory)

        # se abbiamo già una risposta solida, evitiamo una chiamata LLM finale
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
        except Exception as e:
            logger.warning("Final answer LLM failed: %s", e)
            return None

    @staticmethod
    def _is_fallback_good_enough(memory: AgentMemory, fallback: str) -> bool:
        """
        Heuristic: se abbiamo search payload con top result o seller payload chiaro,
        il fallback è già sufficiente e non serve spendere un'altra call LLM.
        """
        if memory.search_payload:
            results = memory.search_payload.get("results") or []
            if results:
                return True

        if memory.seller_payload:
            return True

        if memory.errors and fallback:
            return True

        return False

    @staticmethod
    def _fallback_final_answer(memory: AgentMemory) -> str:
        search_payload = memory.search_payload
        seller_payload = memory.seller_payload

        if search_payload:
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

            intro = f"Il risultato migliore che ho trovato è '{title}'"

            if price is not None:
                intro += f", al prezzo di {price} {currency}"

            if seller_name:
                intro += f", venduto da {seller_name}"

            if trust_score is not None:
                try:
                    intro += f" con trust score {round(float(trust_score) * 100)}%"
                except Exception:
                    pass

            intro += "."

            parts = [intro]

            if analysis:
                parts.append(analysis)

            if seller_payload:
                seller_name = seller_payload.get("seller_name") or memory.last_seller_name or "il venditore"
                count = seller_payload.get("count")
                seller_trust = seller_payload.get("trust_score")
                sentiment = seller_payload.get("sentiment_score")

                seller_sentence = f"Ho anche approfondito {seller_name}"
                details = []

                if seller_trust is not None:
                    try:
                        details.append(f"trust score {round(float(seller_trust) * 100)}%")
                    except Exception:
                        pass

                if sentiment is not None:
                    try:
                        details.append(f"sentiment {round(float(sentiment) * 100)}%")
                    except Exception:
                        pass

                if count is not None:
                    details.append(f"{count} feedback analizzati")

                if details:
                    seller_sentence += " (" + ", ".join(details) + ")"

                seller_sentence += "."
                parts.append(seller_sentence)

            return " ".join(parts)

        if seller_payload:
            seller_name = seller_payload.get("seller_name") or "il venditore"
            count = seller_payload.get("count")
            trust_score = seller_payload.get("trust_score")
            sentiment = seller_payload.get("sentiment_score")

            text = f"Ho analizzato {seller_name}"
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
                details.append(f"{count} feedback totali")

            if details:
                text += " con " + ", ".join(details)

            text += "."
            return text

        if memory.errors:
            return f"Non sono riuscito a completare correttamente l'analisi. Ultimo errore: {memory.errors[-1]}"

        return "Non ho raccolto abbastanza informazioni per produrre una risposta utile."