# Agent Tool-Calling E2E Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, evidence-producing E2E evaluation of MCP tool calling with at least 30 safe scenarios.

**Architecture:** A single pytest module runs the actual `EbayReactAgent` stream for non-mutating scenarios and records its events. A case table provides independently written expectations; event collection produces per-case JSONL records and aggregate metrics. Mutating intents are assessed only through their planned action using a test-local executor guard.

**Tech Stack:** Python, pytest, Pydantic agent events, MCP client, JSONL.

**Spec:** `docs/superpowers/specs/2026-08-30-agent-tool-calling-e2e-design.md`

## Global Constraints

- Execute only with `RUN_AGENT_TOOL_CALLING_E2E=1`.
- Never submit a seller message or mutate a wishlist.
- Store results under `artifacts/evaluations/` and do not require the directory to be committed.
- Treat unavailable optional infrastructure as an explicit skip/report state, never a passing result.

---

### Task 1: Define the evaluator contract and case matrix

**Files:**
- Create: `tests/eval/test_agent_tool_calling_e2e.py`

**Interfaces:**
- Produces: `ToolCallingCase`, `CaseResult`, and `TOOL_CALLING_CASES` (at least 30 cases).

- [ ] **Step 1: Write the failing collection guard test**

```python
def test_e2e_evaluator_requires_explicit_opt_in():
    assert os.getenv("RUN_AGENT_TOOL_CALLING_E2E") == "1"
```

- [ ] **Step 2: Run it to verify it fails without opt-in**

Run: `python -m pytest tests/eval/test_agent_tool_calling_e2e.py -q`

Expected: module skipped instead of accessing LLM, MCP or network.

- [ ] **Step 3: Implement the case dataclasses and table**

```python
@dataclass(frozen=True)
class ToolCallingCase:
    case_id: str
    query: str
    expected_intent: str
    expected_tools: tuple[str, ...]
    mcp_mode: str = "standard"
    execution_mode: Literal["live", "plan_only"] = "live"
```

- [ ] **Step 4: Run collection to verify the opt-in gate remains safe**

Run: `python -m pytest tests/eval/test_agent_tool_calling_e2e.py -q`

Expected: one module-level skip and no service call.

### Task 2: Execute, record and score each case

**Files:**
- Modify: `tests/eval/test_agent_tool_calling_e2e.py`

**Interfaces:**
- Consumes: `ToolCallingCase`.
- Produces: one JSON object per case in `artifacts/evaluations/agent_tool_calling_e2e.jsonl`.

- [ ] **Step 1: Write the failing result-scoring test**

```python
def test_score_marks_missing_expected_tool_as_routing_failure():
    result = score_case(case_with_search_expectation, observed_tools=[])
    assert result.routing_pass is False
```

- [ ] **Step 2: Run the unit test to verify it fails before the scorer exists**

Run: `python -m pytest tests/eval/test_agent_tool_calling_e2e.py::test_score_marks_missing_expected_tool_as_routing_failure -q`

Expected: FAIL because `score_case` is not defined.

- [ ] **Step 3: Implement event collection, safety guard, scoring and JSONL reporting**

```python
async for event in agent.run_stream(request):
    if event["type"] == "tool_start":
        observed_tools.append(event["tool"])
```

- [ ] **Step 4: Run the scoring test to verify it passes**

Run: `python -m pytest tests/eval/test_agent_tool_calling_e2e.py::test_score_marks_missing_expected_tool_as_routing_failure -q`

Expected: PASS.

### Task 3: Verify evaluator usability

**Files:**
- Modify: `tests/eval/test_agent_tool_calling_e2e.py`

- [ ] **Step 1: Run module collection without opt-in**

Run: `python -m pytest tests/eval/test_agent_tool_calling_e2e.py -q`

Expected: skipped module, no external calls.

- [ ] **Step 2: Run static syntax validation**

Run: `python -m py_compile tests/eval/test_agent_tool_calling_e2e.py`

Expected: exit code 0.

- [ ] **Step 3: Run existing deterministic routing regression tests**

Run: `python -m pytest tests/eval/test_agent_routing.py -q`

Expected: all routing tests pass.
