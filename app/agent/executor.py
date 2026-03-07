from __future__ import annotations

from typing import Any, Dict

from app.agent.schemas import Observation, ToolCall
from app.agent.tool_registry import TOOLS, ToolContext


class ToolExecutor:
    def __init__(self, context: ToolContext):
        self.context = context

    def execute(self, tool_call: ToolCall) -> Observation:
        spec = TOOLS.get(tool_call.tool)

        if spec is None:
            return Observation(
                tool=tool_call.tool,
                ok=False,
                error=f"Unknown tool '{tool_call.tool}'",
                summary=f"Tool '{tool_call.tool}' non disponibile.",
            )

        try:
            result = spec.executor(tool_call.input, self.context)

            if not isinstance(result, dict):
                result = {"result": result}

            return Observation(
                tool=tool_call.tool,
                ok=True,
                data=result,
                summary=self._summarize(tool_call.tool, result),
            )

        except Exception as e:
            return Observation(
                tool=tool_call.tool,
                ok=False,
                error=str(e),
                summary=f"{tool_call.tool} failed: {e}",
            )

    def _summarize(self, tool_name: str, data: Dict[str, Any]) -> str:
        if tool_name == "search_pipeline":
            return self._summarize_search(data)

        if tool_name == "seller_pipeline":
            return self._summarize_seller(data)

        return "Tool eseguito."

    @staticmethod
    def _summarize_search(data: Dict[str, Any]) -> str:
        results = data.get("results") or []
        results_count = data.get("results_count", len(results))
        analysis = (data.get("analysis") or "").strip()

        parts = [f"Search completata con {results_count} risultati."]

        top = results[0] if results else None
        if top:
            title = top.get("title")
            price = top.get("price")
            currency = top.get("currency")
            seller = top.get("seller_name") or top.get("seller_username")
            trust = top.get("trust_score")

            if title:
                parts.append(f"Top result: {title}.")
            if price is not None:
                price_text = f"{price} {currency or ''}".strip()
                parts.append(f"Prezzo: {price_text}.")
            if seller:
                parts.append(f"Seller: {seller}.")
            if trust is not None:
                try:
                    parts.append(f"Trust: {round(float(trust) * 100)}%.")
                except Exception:
                    pass

        if analysis:
            parts.append(analysis[:180])

        return " ".join(parts).strip()

    @staticmethod
    def _summarize_seller(data: Dict[str, Any]) -> str:
        seller_name = data.get("seller_name")
        trust_score = data.get("trust_score")
        sentiment_score = data.get("sentiment_score")
        count = data.get("count")

        parts = []

        if seller_name:
            parts.append(f"Analizzato seller {seller_name}.")

        if count is not None:
            parts.append(f"Feedback totali: {count}.")

        if trust_score is not None:
            try:
                parts.append(f"Trust score: {round(float(trust_score) * 100)}%.")
            except Exception:
                pass

        if sentiment_score is not None:
            try:
                parts.append(f"Sentiment score: {round(float(sentiment_score) * 100)}%.")
            except Exception:
                pass

        return " ".join(parts).strip() or "Analisi seller completata."