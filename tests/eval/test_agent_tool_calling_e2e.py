"""Opt-in end-to-end evaluation for MCP tool calling.

Run a complete evaluation only with ``RUN_AGENT_TOOL_CALLING_E2E=1``. The
default engine is ``rule_based`` for repeatable routing; set
``AGENT_E2E_LLM_ENGINE=ollama_cloud`` (or another supported engine) to assess
the LLM planner as well.
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest


REPORT_PATH = Path("artifacts/evaluations/agent_tool_calling_e2e.jsonl")
LIVE_E2E_ENABLED = os.getenv("RUN_AGENT_TOOL_CALLING_E2E") == "1"
STRICT_E2E = os.getenv("AGENT_E2E_STRICT") == "1"
LLM_ENGINE = os.getenv("AGENT_E2E_LLM_ENGINE", "rule_based")
CASE_TIMEOUT_SECONDS = float(os.getenv("AGENT_E2E_CASE_TIMEOUT_SECONDS", "90"))


@dataclass(frozen=True)
class ToolCallingCase:
    case_id: str
    query: str
    expected_intent: str
    expected_tools: tuple[str, ...]
    execution_mode: Literal["live", "plan_only"] = "live"
    mcp_mode: Literal["standard", "playwright_browser"] = "standard"
    requires_playwright: bool = False


@dataclass
class CaseResult:
    case_id: str
    query: str
    expected_intent: str
    expected_tools: list[str]
    observed_tools: list[str] = field(default_factory=list)
    successful_tools: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    final_answer_present: bool = False
    routing_pass: bool = False
    execution_pass: bool | str = False
    completion_pass: bool = False
    status: Literal["pass", "fail", "skip"] = "fail"
    skip_reason: str | None = None
    latency_ms: float = 0.0


# The expected tools are hand-authored contracts, not derived from planner code.
TOOL_CALLING_CASES = (
    ToolCallingCase("search_01", "cerca iPhone 13 ricondizionato", "product_search", ("search_products",)),
    ToolCallingCase("search_02", "laptop gaming sotto 1000 euro", "product_search", ("search_products",)),
    ToolCallingCase("search_03", "cuffie Sony con cancellazione rumore", "product_search", ("search_products",)),
    ToolCallingCase("search_04", "monitor 4k per PS5", "product_search", ("search_products",)),
    ToolCallingCase("search_05", "scarpe Nike numero 42", "product_search", ("search_products",)),
    ToolCallingCase("search_06", "MacBook Air M2 usato", "product_search", ("search_products",)),
    ToolCallingCase("search_07", "fotocamera Canon mirrorless", "product_search", ("search_products",)),
    ToolCallingCase("search_08", "robot aspirapolvere economico", "product_search", ("search_products",)),
    ToolCallingCase("compare_01", "confronta iPhone 15 e Samsung S24", "comparison", ("compare_products",)),
    ToolCallingCase("compare_02", "qual è la differenza tra Steam Deck e ROG Ally", "comparison", ("compare_products",)),
    ToolCallingCase("compare_03", "compara AirPods Pro e Sony WF-1000XM5", "comparison", ("compare_products",)),
    ToolCallingCase("trends_01", "andamento prezzi iPhone 15", "market_trends", ("market_trends",)),
    ToolCallingCase("trends_02", "statistiche mercato smartwatch", "market_trends", ("market_trends",)),
    ToolCallingCase("details_01", "dettagli articolo 123456789012", "item_details", ("get_item_details",)),
    ToolCallingCase("details_02", "voglio dettagli su questo prodotto", "product_search", ("search_products",)),
    ToolCallingCase("shipping_01", "spedizione per articolo 123456789012", "shipping", ("get_shipping_costs",)),
    ToolCallingCase("shipping_02", "quanto costa la spedizione per un iPhone 13", "shipping", ("search_products",)),
    ToolCallingCase("shipping_03", "voglio sapere la spedizione", "shipping", ("search_products",)),
    ToolCallingCase("seller_01", "feedback del venditore pegaso_it", "seller_analysis", ("analyze_seller",)),
    ToolCallingCase("seller_02", "recensioni venditore mediaworld", "seller_analysis", ("analyze_seller",)),
    ToolCallingCase("seller_03", "affidabilità del venditore top_shop", "seller_analysis", ("analyze_seller",)),
    ToolCallingCase("hybrid_01", "cerca prodotti del venditore pegaso_it", "hybrid", ("search_products", "analyze_seller")),
    ToolCallingCase("hybrid_02", "trova offerte dal venditore top_shop", "hybrid", ("search_products", "analyze_seller")),
    ToolCallingCase("deals_01", "offerte eBay per videogiochi", "product_search", ("search_products",)),
    ToolCallingCase("conversation_01", "ciao", "conversation", (), "plan_only"),
    ToolCallingCase("conversation_02", "hello", "conversation", (), "plan_only"),
    ToolCallingCase("wishlist_01", "aggiungi alla wishlist articolo 123456789012", "product_search", ("manage_wishlist",), "plan_only"),
    ToolCallingCase("wishlist_02", "salva questo prodotto nella wishlist", "product_search", ("search_products",), "plan_only"),
    ToolCallingCase("contact_01", "contatta il venditore pegaso_it", "seller_analysis", ("contact_seller",), "plan_only"),
    ToolCallingCase("contact_02", "scrivi un messaggio al seller top_shop", "contact_seller", ("contact_seller",), "plan_only"),
    ToolCallingCase("browser_01", "cerca Nintendo Switch OLED", "product_search", ("browser_navigate",), "live", "playwright_browser", True),
    ToolCallingCase("browser_02", "cerca Dyson V15", "product_search", ("browser_navigate",), "live", "playwright_browser", True),
)


def score_case(
    *,
    expected_tools: tuple[str, ...],
    observed_tools: list[str],
    successful_tools: list[str] | None = None,
    final_answer_present: bool = True,
    execution_mode: Literal["live", "plan_only"] = "live",
) -> dict[str, bool | str]:
    """Score observable agent behavior against a case's independent contract."""
    expected = set(expected_tools)
    observed = set(observed_tools)
    successful = set(successful_tools or [])
    routing_pass = not expected or bool(expected & observed)
    execution_pass: bool | str
    if execution_mode == "plan_only":
        execution_pass = "not_applicable"
    else:
        execution_pass = not expected or bool(expected & successful)
    completion_pass = final_answer_present
    passed = routing_pass and completion_pass and execution_pass in (True, "not_applicable")
    return {
        "routing_pass": routing_pass,
        "execution_pass": execution_pass,
        "completion_pass": completion_pass,
        "status": "pass" if passed else "fail",
    }


def test_score_marks_missing_expected_tool_as_routing_failure():
    result = score_case(expected_tools=("search_products",), observed_tools=[])
    assert result["routing_pass"] is False


def test_case_matrix_contains_at_least_thirty_unique_cases():
    case_ids = [case.case_id for case in TOOL_CALLING_CASES]
    assert len(TOOL_CALLING_CASES) >= 30
    assert len(case_ids) == len(set(case_ids))


def test_read_only_guard_replaces_persistence_hooks(monkeypatch: pytest.MonkeyPatch):
    pipeline = SimpleNamespace(
        _prepare_and_persist_items=lambda *args: (_ for _ in ()).throw(AssertionError("database write hook should be replaced")),
        index_search_items=lambda *args: (_ for _ in ()).throw(AssertionError("qdrant write hook should be replaced")),
    )
    redis = SimpleNamespace(
        set_json=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("redis write hook should be replaced")),
        set_add=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("redis write hook should be replaced")),
    )

    _install_read_only_guards(monkeypatch, pipeline, redis)

    items, persisted = pipeline._prepare_and_persist_items(None, [{"ebay_id": "1"}], {})
    assert items == [{"ebay_id": "1"}]
    assert persisted == 0
    assert pipeline.index_search_items([]) is None
    assert redis.set_json("key", {}) is None
    assert redis.set_add("key", "value") is None


def test_tool_calling_guard_bypasses_task_decomposition(monkeypatch: pytest.MonkeyPatch):
    agent_module = SimpleNamespace(
        decompose_query=lambda *args: (_ for _ in ()).throw(AssertionError("task decomposition should be bypassed"))
    )
    _disable_task_decomposition(monkeypatch, agent_module)
    assert agent_module.decompose_query("cerca cuffie", "rule_based") == []


def _write_report(result: CaseResult) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("a", encoding="utf-8") as report:
        report.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def _install_read_only_guards(
    monkeypatch: pytest.MonkeyPatch,
    search_pipeline: Any,
    redis_client: Any,
) -> None:
    """Preserve live retrieval while disabling evaluator-induced persistence."""
    def keep_items_without_persisting(
        db: Any,
        items: list[dict[str, Any]],
        seller_trust_map: dict[str, float],
    ) -> tuple[list[dict[str, Any]], int]:
        return [dict(item) for item in items], 0

    def no_op(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(search_pipeline, "_prepare_and_persist_items", keep_items_without_persisting)
    monkeypatch.setattr(search_pipeline, "index_search_items", no_op)
    monkeypatch.setattr(redis_client, "set_json", no_op)
    monkeypatch.setattr(redis_client, "set_add", no_op)


def _disable_task_decomposition(monkeypatch: pytest.MonkeyPatch, agent_module: Any) -> None:
    """Keep the evaluation focused on planner-to-tool behavior, not ML task setup."""
    monkeypatch.setattr(agent_module, "decompose_query", lambda query, llm_engine: [])


def _safe_executor_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow real planning but prevent contact and wishlist side effects."""
    from app.agent.executor import ToolExecutor
    from app.agent.schemas import Observation

    async def guarded_execute_many(self: Any, actions: list[Any], parallel: bool = False) -> list[Observation]:
        return [
            Observation(
                tool=action.tool,
                ok=True,
                status="ok",
                summary="Tool execution blocked by E2E safety guard.",
                terminal=True,
            )
            for action in actions
        ]

    monkeypatch.setattr(ToolExecutor, "execute_many", guarded_execute_many)


async def _run_case(case: ToolCallingCase, db: Any, monkeypatch: pytest.MonkeyPatch) -> CaseResult:
    from app.agent import ebay_agent
    from app.agent.schemas import AgentRequest
    from app.db.redis import redis_client
    from app.services import search_pipeline

    _install_read_only_guards(monkeypatch, search_pipeline, redis_client)
    _disable_task_decomposition(monkeypatch, ebay_agent)
    if case.execution_mode == "plan_only":
        _safe_executor_patch(monkeypatch)

    agent = ebay_agent.EbayReactAgent(db=db, mcp_mode=case.mcp_mode, strict_mcp=True)
    result = CaseResult(
        case_id=case.case_id,
        query=case.query,
        expected_intent=case.expected_intent,
        expected_tools=list(case.expected_tools),
    )
    started_at = time.perf_counter()
    try:
        request = AgentRequest(query=case.query, llm_engine=LLM_ENGINE, max_steps=3, return_trace=True)
        async with asyncio.timeout(CASE_TIMEOUT_SECONDS):
            async for event in agent.run_stream(request):
                event_type = event.get("type")
                if event_type == "tool_start":
                    result.observed_tools.append(event["tool"])
                elif event_type == "tool_result" and event.get("ok"):
                    result.successful_tools.append(event["tool"])
                elif event_type == "error":
                    result.errors.append(str(event.get("message", "Unknown agent error")))
                elif event_type == "final":
                    result.final_answer_present = bool(str(event.get("final_answer", "")).strip())
    except TimeoutError:
        result.errors.append(f"Timed out after {CASE_TIMEOUT_SECONDS:.0f}s without a complete agent result.")
    except Exception as exc:  # Preserve the failure as evaluation evidence.
        result.errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        result.latency_ms = round((time.perf_counter() - started_at) * 1000, 2)

    score = score_case(
        expected_tools=case.expected_tools,
        observed_tools=result.observed_tools,
        successful_tools=result.successful_tools,
        final_answer_present=result.final_answer_present,
        execution_mode=case.execution_mode,
    )
    result.routing_pass = bool(score["routing_pass"])
    result.execution_pass = score["execution_pass"]
    result.completion_pass = bool(score["completion_pass"])
    result.status = str(score["status"])
    _write_report(result)
    return result


@pytest.fixture(scope="module")
def e2e_db() -> Any:
    if not LIVE_E2E_ENABLED:
        pytest.skip("Set RUN_AGENT_TOOL_CALLING_E2E=1 to run live MCP/LLM evaluation.")
    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("DATABASE_URL"):
        pytest.skip("DATABASE_URL is required for the agent E2E evaluation.")

    from sqlalchemy import text
    from app.db.database import SessionLocal

    db = SessionLocal()
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:
        db.close()
        pytest.skip(f"PostgreSQL is unavailable: {exc}")
    yield db
    db.close()


@pytest.mark.parametrize("case", TOOL_CALLING_CASES, ids=lambda case: case.case_id)
def test_agent_tool_calling_end_to_end(
    case: ToolCallingCase,
    e2e_db: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if case.requires_playwright and os.getenv("RUN_PLAYWRIGHT_E2E") != "1":
        pytest.skip("Set RUN_PLAYWRIGHT_E2E=1 to include browser-mode cases.")

    result = asyncio.run(_run_case(case, e2e_db, monkeypatch))
    print(json.dumps(asdict(result), ensure_ascii=False))
    if STRICT_E2E:
        assert result.status == "pass", result


@pytest.fixture(scope="module", autouse=True)
def write_evaluation_summary() -> Any:
    if not LIVE_E2E_ENABLED:
        yield
        return

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("", encoding="utf-8")
    yield
    rows = [json.loads(line) for line in REPORT_PATH.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        return
    live_rows = [row for row in rows if row["status"] != "skip"]
    latencies = [row["latency_ms"] for row in live_rows]
    summary = {
        "summary": True,
        "cases": len(rows),
        "pass": sum(row["status"] == "pass" for row in rows),
        "fail": sum(row["status"] == "fail" for row in rows),
        "routing_rate": round(sum(row["routing_pass"] for row in live_rows) / len(live_rows), 3) if live_rows else 0,
        "execution_rate": round(
            sum(row["execution_pass"] in (True, "not_applicable") for row in live_rows) / len(live_rows), 3
        ) if live_rows else 0,
        "median_latency_ms": round(statistics.median(latencies), 2) if latencies else 0,
    }
    with REPORT_PATH.open("a", encoding="utf-8") as report:
        report.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print("[agent-tool-calling-e2e] " + json.dumps(summary, ensure_ascii=False))
