"""
Regressione: un turno con immagine non deve MAI degradare silenziosamente
in una risposta costruita sullo storico della conversazione.

Prima del fix, se `describe_image_with_vision` restituiva None, `request.query`
restava "" e il planner ripiegava sull'intent "conversation", che risponde
usando la conversation_history: l'utente riceveva la risposta della richiesta
testuale PRECEDENTE.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import app.llm.vision as vision_module
from app.agent.ebay_agent import EbayReactAgent
from app.agent.schemas import AgentRequest


class _MemoryServiceSpy:
    """Sostituisce MemoryService per non toccare Redis nei test."""

    def __init__(self) -> None:
        self.cleared: List[Any] = []

    def clear_vision_description(self, user: Any) -> None:
        self.cleared.append(user)


def _drain(agent: EbayReactAgent, request: AgentRequest, stop_after: int) -> List[Dict[str, Any]]:
    """Consuma al massimo `stop_after` eventi, poi abbandona il generatore.

    Serve a fermarsi prima che run_stream tocchi DB/MCP, che qui non esistono.
    """

    async def _run() -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        stream = agent.run_stream(request)
        try:
            async for event in stream:
                events.append(event)
                if len(events) >= stop_after:
                    break
        finally:
            await stream.aclose()
        return events

    return asyncio.run(_run())


def test_vision_failure_emits_error_and_stops(monkeypatch):
    """Vision KO -> evento error esplicito, nessun fallback su conversation."""

    async def _fails(_image: str):
        return None

    monkeypatch.setattr(vision_module, "describe_image_with_vision", _fails)

    agent = EbayReactAgent(db=None, user=None, prefer_mcp=False)
    spy = _MemoryServiceSpy()
    agent.memory_service = spy

    request = AgentRequest(query="", image="data:image/png;base64,AAAA", max_steps=1)
    events = _drain(agent, request, stop_after=5)

    types = [e.get("type") for e in events]
    assert "error" in types, f"atteso un evento error, ottenuto {types}"

    error_event = next(e for e in events if e.get("type") == "error")
    assert "immagine" in error_event["message"].lower()

    # Lo stream si ferma: nessun evento di planning dopo l'errore.
    assert types[-1] == "error", types
    # La descrizione della foto precedente viene azzerata.
    assert spy.cleared == [None]


def test_vision_without_brand_or_tags_still_builds_a_real_query(monkeypatch):
    """Fallback su JSON non parsabile: la query non deve ridursi a 'Trova:'."""

    async def _returns_description_only(_image: str):
        return {
            "description": "Scarpa da ginnastica bianca con suola spessa e lacci piatti",
            "tags": [],
            "brand": None,
            "condition_clues": None,
            "confidence": 0.5,
        }

    monkeypatch.setattr(vision_module, "describe_image_with_vision", _returns_description_only)

    agent = EbayReactAgent(db=None, user=None, prefer_mcp=False)
    agent.memory_service = _MemoryServiceSpy()

    request = AgentRequest(query="", image="data:image/png;base64,AAAA", max_steps=1)
    events = _drain(agent, request, stop_after=2)

    types = [e.get("type") for e in events]
    assert "vision_analysis" in types, types
    assert "error" not in types, types

    assert request.query.strip() not in {"", "Trova:", "Trova"}
    assert "scarpa" in request.query.lower()


def test_vision_success_appends_compact_tags_to_user_text(monkeypatch):
    """Con testo utente + vision, la query resta il testo arricchito dai tag."""

    async def _ok(_image: str):
        return {
            "description": "Sneakers bianche",
            "tags": ["sneakers", "bianche", "pelle", "lacci", "extra-ignorato"],
            "brand": "Nike",
            "condition_clues": None,
            "confidence": 0.9,
        }

    monkeypatch.setattr(vision_module, "describe_image_with_vision", _ok)

    agent = EbayReactAgent(db=None, user=None, prefer_mcp=False)
    agent.memory_service = _MemoryServiceSpy()

    request = AgentRequest(query="sotto i 100 euro", image="data:image/png;base64,AAAA", max_steps=1)
    _drain(agent, request, stop_after=2)

    assert request.query.startswith("sotto i 100 euro (")
    assert "Nike" in request.query
    # Solo i primi 4 tag finiscono nella query.
    assert "extra-ignorato" not in request.query


def test_vision_with_no_usable_output_stops_instead_of_querying_nothing(monkeypatch):
    """Vision risponde ma senza brand, tag né descrizione: niente query vuota."""

    async def _empty(_image: str):
        return {
            "description": "",
            "tags": [],
            "brand": None,
            "condition_clues": None,
            "confidence": 0.1,
        }

    monkeypatch.setattr(vision_module, "describe_image_with_vision", _empty)

    agent = EbayReactAgent(db=None, user=None, prefer_mcp=False)
    spy = _MemoryServiceSpy()
    agent.memory_service = spy

    request = AgentRequest(query="", image="data:image/png;base64,AAAA", max_steps=1)
    events = _drain(agent, request, stop_after=5)

    types = [e.get("type") for e in events]
    assert types[-1] == "error", types
    assert request.query.strip() == ""
    assert spy.cleared == [None]


def test_vision_with_no_usable_output_keeps_user_text(monkeypatch):
    """Stesso caso ma con testo utente: si prosegue con la sola query testuale."""

    async def _empty(_image: str):
        return {
            "description": "",
            "tags": [],
            "brand": None,
            "condition_clues": None,
            "confidence": 0.1,
        }

    monkeypatch.setattr(vision_module, "describe_image_with_vision", _empty)

    agent = EbayReactAgent(db=None, user=None, prefer_mcp=False)
    agent.memory_service = _MemoryServiceSpy()

    request = AgentRequest(query="zaino nero", image="data:image/png;base64,AAAA", max_steps=1)
    events = _drain(agent, request, stop_after=2)

    types = [e.get("type") for e in events]
    assert "error" not in types, types
    assert request.query == "zaino nero"
