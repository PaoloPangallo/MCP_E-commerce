# Fix contact_seller_playwright Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the `contact_seller_playwright` flow so the agent correctly sends a human-readable message to an eBay seller using the real Chrome profile, extracting the seller from browser navigation state when needed.

**Architecture:** `contact_seller_playwright` intentionally uses a **separate Chrome instance** (real user profile with eBay login) while `BrowserManager` handles anonymous browsing. The planner bridges the two by extracting seller/URL from the active browser state and generating a proper message via LLM. No merging of the two systems.

**Tech Stack:** Python, FastMCP, Playwright (async), `call_llm`, `parse_query_service`, `AgentMemory.tool_states`

---

## Root Causes

| # | Bug | Location |
|---|-----|----------|
| 1 | Raw `user_query` sent as seller message | `planner.py:530` |
| 2 | URL fallback reads `search_payload`/`top_results` (always empty in playwright mode) | `planner.py:516–527` |
| 3 | Seller name never extracted from BrowserManager page observations | `planner.py` / `executor.py` |
| 4 | No `last_seller_name` populated when agent views a product via browser tools | `memory.py` |

---

## File Map

| File | What changes |
|------|-------------|
| `app/agent/planner.py` | Fix `_normalize_action_input` for `contact_seller_playwright` (bugs 1, 2); add `_generate_seller_message()` async method |
| `app/agent/memory.py` | Populate `last_seller_name` from browser_navigate/browser_type observations |
| `app/agent/executor.py` | Call `_extract_seller_from_browser_result()` after browser tool success |

---

## Task 1 — Fix URL fallback to use browser navigation state

**Files:**
- Modify: `app/agent/planner.py:515–527`

The fallback at lines 516–527 reads `search_payload` and `top_results`, which are populated only by `search_products` — a tool excluded in playwright mode. In playwright mode the browser state is stored in `memory.tool_states["browser_type"]["data"]` and `memory.tool_states["browser_navigate"]["data"]`.

- [ ] **Step 1: Read the current fallback block**

File: `app/agent/planner.py`, lines 502–532. Current code:
```python
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
            # Nessun seller noto — fallback a URL prodotto dal scratchpad corrente
            scratchpad = memory.scratchpad()
            top_results = scratchpad.get("top_results") or []
            if top_results and top_results[0].get("url"):
                action_input["product_url"] = top_results[0]["url"]
            else:
                search = memory.search_payload or {}
                results = search.get("results") or []
                if results and results[0].get("url"):
                    action_input["product_url"] = results[0]["url"]
                else:
                    return None

    if not action_input.get("message"):
        action_input["message"] = memory.user_query

    return action_input
```

- [ ] **Step 2: Replace with browser-state-aware fallback**

Replace the entire `elif action == "contact_seller_playwright":` block (lines 502–532) with:

```python
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
            # In playwright mode: look at the current browser page URL from BrowserManager
            # Tool states store the last browser result which includes the current URL
            browser_url = None
            for browser_tool in ("browser_type", "browser_navigate", "browser_get_view"):
                state_data = (memory.tool_states.get(browser_tool) or {}).get("data") or {}
                url = state_data.get("url", "")
                if url and "ebay" in url.lower():
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
        action_input["message"] = await self._generate_seller_message(memory)

    return action_input
```

> Note: `_normalize_action_input` must become `async` for this step. See Task 2.

- [ ] **Step 3: Make `_normalize_action_input` async**

The method signature at line 378 currently is:
```python
def _normalize_action_input(
        self,
        action: str,
        action_input: Dict[str, Any],
        memory: AgentMemory,
) -> Optional[Dict[str, Any]]:
```

Change to:
```python
async def _normalize_action_input(
        self,
        action: str,
        action_input: Dict[str, Any],
        memory: AgentMemory,
) -> Optional[Dict[str, Any]]:
```

- [ ] **Step 4: Update all callers of `_normalize_action_input` to use `await`**

Search in `planner.py` for all calls to `self._normalize_action_input(` and add `await`:

```bash
grep -n "_normalize_action_input" app/agent/planner.py
```

Each call like:
```python
normalized_input = self._normalize_action_input(action, action_input, memory)
```
becomes:
```python
normalized_input = await self._normalize_action_input(action, action_input, memory)
```

Also update call in `_safe_fallback_decide` (line 313):
```python
normalized_input = await self._normalize_action_input(tool_name, {}, memory)
```
But `_safe_fallback_decide` is currently **sync** — it must become async too. Update its signature:
```python
async def _safe_fallback_decide(
        self,
        memory: AgentMemory,
        forced_intent: Optional[str] = None,
) -> PlannerOutput:
```

And update its caller in `decide()` (line 84):
```python
return await self._safe_fallback_decide(memory)
```

And in `_llm_decide` around lines 258–262:
```python
return await self._safe_fallback_decide(memory, forced_intent=intent)
# and
return await self._safe_fallback_decide(memory, forced_intent=intent)
```

And in `_decide_from_task_queue` (line 161):
```python
action_input = await self._normalize_action_input(tool, task.get("input") or {}, memory)
```
Making `_decide_from_task_queue` async too, and updating its callers in `decide()` and `_safe_fallback_decide` with `await`.

- [ ] **Step 5: Verify app starts without errors**

```bash
.\.venv\Scripts\python.exe -c "from app.agent.planner import ReactPlanner; print('OK')"
```
Expected: `OK`

---

## Task 2 — Add `_generate_seller_message()` LLM method

**Files:**
- Modify: `app/agent/planner.py` (new method)

The current default `action_input["message"] = memory.user_query` sends the raw user prompt ("contatta il venditore di questo iPhone e chiedi...") as the actual message to the seller. This is unacceptable. We need a short, professional message extracted from the user's intent.

- [ ] **Step 1: Add method to ReactPlanner class**

Add this method after `_precompute_browser_query` (around line 128):

```python
async def _generate_seller_message(self, memory: AgentMemory) -> str:
    """Generate a short, professional message to send to an eBay seller.

    Extracts the user's actual intent from their query and formats it as
    a polite seller message. Falls back to a generic inquiry if LLM fails.
    """
    user_query = (memory.user_query or "").strip()
    if not user_query:
        return "Salve, volevo richiedere informazioni su questo articolo. Grazie."

    prompt = (
        "L'utente vuole inviare un messaggio a un venditore eBay. "
        f"La sua richiesta è: \"{user_query}\"\n\n"
        "Scrivi SOLO il testo del messaggio da inviare al venditore: breve, educato, professionale. "
        "Non includere saluti ridondanti. Non includere spiegazioni. Solo il testo del messaggio. "
        "Massimo 3 frasi."
    )
    try:
        message, _ = await call_llm(prompt)
        return (message or "").strip() or "Salve, volevo richiedere informazioni su questo articolo. Grazie."
    except Exception:
        return "Salve, volevo richiedere informazioni su questo articolo. Grazie."
```

- [ ] **Step 2: Verify `call_llm` is already imported at the top of planner.py**

```bash
grep -n "from app.llm" app/agent/planner.py | head -5
```

If not present, add at the top of the imports section:
```python
from app.llm.client import call_llm
```

- [ ] **Step 3: Quick smoke test**

```bash
.\.venv\Scripts\python.exe -c "
import asyncio
from app.agent.planner import ReactPlanner
from app.agent.memory import AgentMemory
p = ReactPlanner(mcp_client=None)
mem = AgentMemory.__new__(AgentMemory)
mem.user_query = 'contatta il venditore e chiedi se hanno ancora disponibilità'
msg = asyncio.run(p._generate_seller_message(mem))
print('Message:', msg)
"
```
Expected: a short Italian professional message, not the raw query.

---

## Task 3 — Populate `last_seller_name` from browser page observations

**Files:**
- Modify: `app/agent/memory.py` — `add_observation()` method (around line 500)

When the BrowserManager navigates to an eBay product listing, the page_text in the observation summary often contains seller info (e.g., "Venduto da: jjtech2020"). We should extract and store this automatically.

- [ ] **Step 1: Add seller extraction helper**

Add at the module level of `memory.py` (near the top, after imports):

```python
import re as _re

def _extract_seller_from_page_text(page_text: str) -> Optional[str]:
    """Extract eBay seller username from browser page text observation."""
    if not page_text:
        return None
    # Pattern: "Venduto da: username" or "Seller: username" or "da username"
    patterns = [
        r"[Vv]enduto\s+da[:\s]+([A-Za-z0-9_\-\.]{3,})",
        r"[Ss]eller[:\s]+([A-Za-z0-9_\-\.]{3,})",
        r"[Vv]enditore[:\s]+([A-Za-z0-9_\-\.]{3,})",
    ]
    for pattern in patterns:
        match = _re.search(pattern, page_text)
        if match:
            candidate = match.group(1).strip()
            # Reject common false positives
            if candidate.lower() not in {"ebay", "paypal", "feedback", "contatta", "italy"}:
                return candidate
    return None
```

- [ ] **Step 2: Call it in `add_observation` after browser tool success**

In the `add_observation` method (around line 504), add after the `tool_states` assignment block:

```python
# Auto-populate last_seller_name from browser page observations
if observation.tool in {"browser_navigate", "browser_type", "browser_get_view"} and observation.ok:
    page_text = (observation.data or {}).get("page_text", "")
    seller = _extract_seller_from_page_text(page_text)
    if seller and not self.last_seller_name:
        self.last_seller_name = seller
        logger.debug("memory: extracted seller_name=%s from %s observation", seller, observation.tool)
```

- [ ] **Step 3: Verify no import errors**

```bash
.\.venv\Scripts\python.exe -c "from app.agent.memory import AgentMemory; print('OK')"
```
Expected: `OK`

---

## Task 4 — Verify end-to-end contact flow

- [ ] **Step 1: Start the application**

```powershell
.\start_dev.ps1
```

- [ ] **Step 2: Make sure Chrome is running with CDP enabled (optional, for fastest path)**

If Chrome is running normally (no CDP), the tool will launch a new Chrome window with the user's profile. That's fine.

- [ ] **Step 3: Send a contact request in playwright mode**

POST to `/agent/stream`:
```json
{
  "query": "contatta il venditore jjtech2020 e chiedi se hanno scorte disponibili",
  "mcp_mode": "playwright_browser",
  "llm_engine": "ollama_cloud"
}
```

Expected log sequence:
```
INFO: planner: contact_seller intent detected
INFO: contact_seller_playwright: using IntermediatedFAQ URL for seller=jjtech2020
INFO: MCP: contact_seller_playwright running
INFO: _generate_seller_message: generated "Salve jjtech2020..."
INFO: ToolExecutor MCP success | tool=contact_seller_playwright | contact_status=message_sent
```

- [ ] **Step 4: Test fallback — no seller name in query, but browser has eBay page open**

POST:
```json
{
  "query": "contatta il venditore di questo prodotto",
  "mcp_mode": "playwright_browser"
}
```

Expected: agent uses `browser_url` from `memory.tool_states["browser_navigate"]["data"]["url"]` as `product_url`.

- [ ] **Step 5: Verify message is not raw query**

In the logs, look for:
```
INFO: contact_seller_playwright running | product_url=... | message=Salve, ...
```
The message should NOT be "contatta il venditore di questo prodotto".

---

## Known Limitations (Out of Scope)

- **eBay UI changes**: The selectors in `playwright_contact.py` (lines 95–141) may break if eBay redesigns the contact form. No fix planned — maintain as-is.
- **Login expiry**: If Chrome cookies expired, the tool returns `login_required`. User must manually log in. No automatic re-auth planned.
- **Dual-browser UX**: The agent uses BrowserManager for search (Chromium, headless-ish) and `contact_seller_playwright` uses real Chrome for auth. Two browser windows may appear. This is intentional — real Chrome carries eBay login cookies.
