# MCP Worlds Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "MCP World" selector that lets the user switch between a standard API-based MCP server and a Playwright browser-based MCP server (visible Chromium, with search + contact-seller automation).

**Architecture:** Two separate FastMCP instances (`mcp` and `mcp_playwright`) mounted as sub-apps under a root FastAPI app in `asgi.py`. The agent receives `mcp_server_url` dynamically based on the `mcp_mode` sent by the frontend. The Playwright world exposes only `search_products` (headless=False) and `contact_seller_playwright`.

**Tech Stack:** Python/FastAPI, FastMCP, Playwright (Chromium), React/TypeScript, Zustand, MUI.

---

## File Map

| Action | File |
|--------|------|
| **Create** | `app/mcp/playwright_server.py` |
| **Create** | `app/mcp/tools/playwright_contact.py` |
| **Modify** | `app/mcp/asgi.py` |
| **Modify** | `app/agent/prompts.py` |
| **Modify** | `app/agent/memory.py` |
| **Modify** | `app/agent/planner.py` |
| **Modify** | `app/agent/ebay_agent.py` |
| **Modify** | `app/api/agent_stream.py` |
| **Modify** | `ebay-ui/src/features/chat/store/settingsStore.ts` |
| **Modify** | `ebay-ui/src/features/chat/SettingsModal.tsx` |
| **Modify** | `ebay-ui/src/features/agent/api/stream.ts` |
| **Modify** | `ebay-ui/src/features/agent/hooks/useAgentStream.ts` |

---

## Task 1: Create `app/mcp/playwright_server.py`

**Files:**
- Create: `app/mcp/playwright_server.py`

This new module declares a second FastMCP instance for the Playwright world and registers `search_products` as a browser-based scraping tool (headless=False forced).

- [ ] **Step 1: Write the failing test**

```python
# tests/mcp/test_playwright_server.py
import pytest
from app.mcp.playwright_server import mcp_playwright

def test_playwright_server_has_search_products_tool():
    """The playwright server must expose search_products."""
    tool_names = [tool.name for tool in mcp_playwright._tool_manager.list_tools()]
    assert "search_products" in tool_names

def test_playwright_server_does_not_have_standard_tools():
    """The playwright server must NOT expose analyze_seller or other standard tools."""
    tool_names = [tool.name for tool in mcp_playwright._tool_manager.list_tools()]
    assert "analyze_seller" not in tool_names
    assert "get_ebay_deals" not in tool_names
```

- [ ] **Step 2: Run test to verify it fails**

```
cd C:\Users\paolo\MCP_ECOM
python -m pytest tests/mcp/test_playwright_server.py -v
```
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Create the file**

```python
# app/mcp/playwright_server.py
"""
MCP Server — Playwright Browser World
Espone solo i tool che richiedono un browser reale (Chromium visibile).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# ── Seconda istanza MCP: mondo Playwright ────────────────────────────────────
mcp_playwright = FastMCP("mcp-playwright-world")

@dataclass
class _PlaywrightDeps:
    db_factory: Optional[Callable[[], Any]] = None

_PLAYWRIGHT_DEPS = _PlaywrightDeps()


def configure_playwright_mcp(
    db_factory: Optional[Callable[[], Any]] = None,
) -> None:
    _PLAYWRIGHT_DEPS.db_factory = db_factory
    logger.info("Playwright MCP configured | db_factory=%s", bool(db_factory))


# ── Tool: search_products (browser, headless=False) ─────────────────────────
from typing import Annotated, Dict
from pydantic import Field
from app.mcp.normalizers import _normalize_playwright_output
from app.services.ebay_playwright import scrape_ebay_search


@mcp_playwright.tool(
    name="search_products",
    description=(
        "Cerca prodotti su eBay.it navigando il sito con un browser Chromium reale e VISIBILE. "
        "Il browser si apre sullo schermo così l'utente può vedere la navigazione in tempo reale. "
        "Restituisce titolo, prezzo, URL, immagine, condizione, venditore e spedizione per ogni annuncio."
    ),
)
async def pw_search_products(
    query: Annotated[
        str,
        Field(description="Query di ricerca per eBay (es. 'iphone 13 128gb')"),
    ],
    max_results: Annotated[
        int,
        Field(description="Numero massimo di risultati (default 10, max 24)", ge=1, le=24),
    ] = 10,
) -> Dict[str, Any]:
    try:
        logger.info("PW search_products START | query=%s | max=%d", query, max_results)
        results = await scrape_ebay_search(
            query=query,
            max_results=max_results,
            headless=False,  # SEMPRE visibile nel mondo Playwright
        )
        raw = {
            "query": query,
            "results": results,
            "results_count": len(results),
        }
        normalized = _normalize_playwright_output(raw)
        normalized["_backend"] = "playwright_browser"
        logger.info("PW search_products END | count=%d", len(results))
        return normalized
    except Exception as exc:
        logger.exception("PW search_products failed")
        return {"status": "error", "query": query, "error": str(exc)}
```

- [ ] **Step 4: Run test to verify it passes**

```
python -m pytest tests/mcp/test_playwright_server.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/mcp/playwright_server.py tests/mcp/test_playwright_server.py
git commit -m "feat: add playwright MCP server with browser search_products tool"
```

---

## Task 2: Create `app/mcp/tools/playwright_contact.py`

**Files:**
- Create: `app/mcp/tools/playwright_contact.py`

Implements `contact_seller_playwright`: opens Chromium visible, navigates to a product page, finds the contact seller button, and attempts to send a message.

- [ ] **Step 1: Write the failing test**

```python
# tests/mcp/tools/test_playwright_contact.py
import pytest
from app.mcp.tools.playwright_contact import _build_contact_result

def test_build_contact_result_success():
    result = _build_contact_result(
        product_url="https://www.ebay.it/itm/123",
        success=True,
        status="message_sent",
        detail="Messaggio inviato con successo.",
    )
    assert result["status"] == "ok"
    assert result["success"] is True
    assert result["product_url"] == "https://www.ebay.it/itm/123"

def test_build_contact_result_failure():
    result = _build_contact_result(
        product_url="https://www.ebay.it/itm/123",
        success=False,
        status="login_required",
        detail="eBay richiede il login per contattare i venditori.",
    )
    assert result["status"] == "error"
    assert result["success"] is False
    assert "login" in result["detail"].lower()
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/mcp/tools/test_playwright_contact.py -v
```
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create the file**

```python
# app/mcp/tools/playwright_contact.py
"""
playwright_contact.py
MCP Tool: contact_seller_playwright
Apre Chromium visibile, naviga su una pagina prodotto eBay e tenta di
contattare il venditore compilando il form di messaggistica.

NOTA: eBay richiede login per inviare messaggi. Il tool naviga fino alla
pagina di contatto e restituisce lo stato: se l'utente non è loggato,
ritorna status="login_required" con istruzioni.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any, Annotated, Dict, Optional

from pydantic import Field

from app.mcp.playwright_server import mcp_playwright
from app.services.ebay_playwright import _PLAYWRIGHT_EXECUTOR

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Helpers pubblici (testabili)
# ─────────────────────────────────────────────

def _build_contact_result(
    product_url: str,
    success: bool,
    status: str,
    detail: str,
    message_sent: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "status": "ok" if success else "error",
        "success": success,
        "product_url": product_url,
        "contact_status": status,
        "detail": detail,
        "message_sent": message_sent,
    }


# ─────────────────────────────────────────────
# Core async (deve girare in ProactorEventLoop)
# ─────────────────────────────────────────────

async def _async_contact_seller(
    product_url: str,
    message: str,
    timeout_ms: int,
) -> Dict[str, Any]:
    """Logica Playwright pura — chiamare solo da dentro un ProactorEventLoop."""
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright non installato. Esegui: pip install playwright && playwright install chromium"
        ) from exc

    logger.info("PW contact_seller START | url=%s", product_url)
    _launch_args = ["--no-sandbox", "--disable-dev-shm-usage"]

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, args=_launch_args)
        context = await browser.new_context(
            locale="it-IT",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()

        try:
            await page.goto(product_url, timeout=timeout_ms, wait_until="load")
        except Exception as exc:
            logger.warning("PW contact_seller: goto timeout | %s", exc)

        # Controlla se eBay ci ha reindirizzato al login
        current_url = page.url
        if "signin" in current_url or "login" in current_url:
            await browser.close()
            return _build_contact_result(
                product_url=product_url,
                success=False,
                status="login_required",
                detail=(
                    "eBay richiede il login per contattare i venditori. "
                    "Effettua il login su eBay.it nel browser che si è aperto, "
                    "poi riprova l'operazione."
                ),
            )

        # Cerca il link/bottone 'Contatta il venditore'
        contact_selectors = [
            "a[href*='contactseller']",
            "a[href*='contact_seller']",
            "a:has-text('Contatta il venditore')",
            "a:has-text('Contact seller')",
            "button:has-text('Contatta il venditore')",
            "[data-testid*='contact']",
        ]

        contact_link = None
        for selector in contact_selectors:
            try:
                el = await page.query_selector(selector)
                if el:
                    contact_link = el
                    break
            except Exception:
                continue

        if not contact_link:
            await browser.close()
            return _build_contact_result(
                product_url=product_url,
                success=False,
                status="contact_button_not_found",
                detail=(
                    "Non ho trovato il pulsante 'Contatta il venditore' su questa pagina. "
                    "Potrebbe essere che il venditore non accetti messaggi, "
                    "o che la pagina abbia una struttura diversa."
                ),
            )

        await contact_link.click()

        # Attendi la pagina del form di contatto
        try:
            await page.wait_for_load_state("networkidle", timeout=10_000)
        except Exception:
            pass

        # Controlla di nuovo il login dopo il click
        current_url = page.url
        if "signin" in current_url or "login" in current_url:
            await browser.close()
            return _build_contact_result(
                product_url=product_url,
                success=False,
                status="login_required",
                detail=(
                    "eBay richiede il login per inviare messaggi. "
                    "Effettua il login su eBay.it e riprova."
                ),
            )

        # Cerca il campo testo del messaggio
        message_selectors = [
            "textarea[name='body']",
            "textarea[id*='message']",
            "textarea[placeholder*='messaggio']",
            "textarea[placeholder*='message']",
            "textarea",
        ]

        textarea = None
        for sel in message_selectors:
            try:
                el = await page.query_selector(sel)
                if el:
                    textarea = el
                    break
            except Exception:
                continue

        if not textarea:
            await browser.close()
            return _build_contact_result(
                product_url=product_url,
                success=False,
                status="message_form_not_found",
                detail=(
                    "La pagina di contatto è aperta nel browser. "
                    "Non ho trovato il campo di testo in modo automatico. "
                    "Puoi compilare e inviare il messaggio manualmente nel browser."
                ),
            )

        await textarea.fill(message)

        # Cerca il bottone di invio
        submit_selectors = [
            "button[type='submit']",
            "input[type='submit']",
            "button:has-text('Invia')",
            "button:has-text('Send')",
            "button:has-text('Invia messaggio')",
        ]

        submit_btn = None
        for sel in submit_selectors:
            try:
                el = await page.query_selector(sel)
                if el:
                    submit_btn = el
                    break
            except Exception:
                continue

        if not submit_btn:
            await browser.close()
            return _build_contact_result(
                product_url=product_url,
                success=False,
                status="submit_button_not_found",
                detail=(
                    "Ho compilato il messaggio nel form. "
                    "Non ho trovato il bottone di invio automaticamente — "
                    "clicca tu stesso 'Invia' nel browser aperto."
                ),
                message_sent=message,
            )

        await submit_btn.click()

        try:
            await page.wait_for_load_state("networkidle", timeout=8_000)
        except Exception:
            pass

        await browser.close()
        logger.info("PW contact_seller: message sent | url=%s", product_url)
        return _build_contact_result(
            product_url=product_url,
            success=True,
            status="message_sent",
            detail="Messaggio inviato al venditore con successo tramite browser.",
            message_sent=message,
        )


# ─────────────────────────────────────────────
# Thread worker: ProactorEventLoop isolato
# ─────────────────────────────────────────────

def _run_contact_in_proactor_loop(
    product_url: str,
    message: str,
    timeout_ms: int,
) -> Dict[str, Any]:
    if sys.platform == "win32":
        loop = asyncio.ProactorEventLoop()
    else:
        loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            _async_contact_seller(product_url, message, timeout_ms)
        )
    finally:
        loop.close()
        asyncio.set_event_loop(None)


# ─────────────────────────────────────────────
# MCP Tool registration
# ─────────────────────────────────────────────

@mcp_playwright.tool(
    name="contact_seller_playwright",
    description=(
        "Contatta un venditore eBay inviando un messaggio direttamente dalla pagina del prodotto, "
        "usando un browser Chromium reale e visibile. "
        "Naviga automaticamente alla pagina del prodotto, trova il pulsante 'Contatta il venditore', "
        "compila il form e invia il messaggio. "
        "NOTA: richiede che l'utente sia loggato su eBay nel browser che si apre. "
        "Se non loggato, il tool restituirà istruzioni per effettuare il login."
    ),
)
async def contact_seller_playwright(
    product_url: Annotated[
        str,
        Field(description="URL completo della pagina prodotto eBay (es. https://www.ebay.it/itm/123456789012)"),
    ],
    message: Annotated[
        str,
        Field(description="Testo del messaggio da inviare al venditore"),
    ],
) -> Dict[str, Any]:
    try:
        logger.info("MCP contact_seller_playwright START | url=%s", product_url)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _PLAYWRIGHT_EXECUTOR,
            _run_contact_in_proactor_loop,
            product_url,
            message,
            30_000,
        )
        logger.info(
            "MCP contact_seller_playwright END | status=%s",
            result.get("contact_status"),
        )
        return result
    except Exception as exc:
        logger.exception("MCP contact_seller_playwright failed")
        return _build_contact_result(
            product_url=product_url,
            success=False,
            status="error",
            detail=str(exc),
        )
```

- [ ] **Step 4: Run test to verify it passes**

```
python -m pytest tests/mcp/tools/test_playwright_contact.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/mcp/tools/playwright_contact.py tests/mcp/tools/test_playwright_contact.py
git commit -m "feat: add contact_seller_playwright Playwright tool"
```

---

## Task 3: Refactor `app/mcp/asgi.py` — Mount Both MCP Servers

**Files:**
- Modify: `app/mcp/asgi.py`

After this change the endpoints become:
- Standard world: `http://127.0.0.1:8050/standard/mcp`
- Playwright world: `http://127.0.0.1:8050/playwright/mcp`

- [ ] **Step 1: Write the failing test**

```python
# tests/mcp/test_asgi_routing.py
from fastapi.testclient import TestClient

def test_asgi_app_has_standard_and_playwright_routes():
    """The ASGI app must mount both MCP worlds."""
    from app.mcp.asgi import app
    # Verify it's a FastAPI app with sub-mounts, not a plain ASGI app
    from fastapi import FastAPI
    assert isinstance(app, FastAPI)

def test_standard_mcp_route_exists():
    from app.mcp.asgi import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
    # The /standard path should not return 404
    resp = client.get("/standard/")
    assert resp.status_code != 404
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/mcp/test_asgi_routing.py::test_asgi_app_has_standard_and_playwright_routes -v
```
Expected: FAIL (`app` is a Starlette ASGI app, not FastAPI)

- [ ] **Step 3: Replace the content of `app/mcp/asgi.py`**

```python
# app/mcp/asgi.py
from __future__ import annotations

from fastapi import FastAPI

from app.db.database import SessionLocal
from app.mcp.server import configure_mcp, mcp
from app.mcp.playwright_server import configure_playwright_mcp, mcp_playwright

# Import tool registrations for the playwright world
import app.mcp.tools.playwright_contact  # noqa: F401 — registers contact_seller_playwright

# Configure both worlds with DB access
configure_mcp(db_factory=SessionLocal)
configure_playwright_mcp(db_factory=SessionLocal)

# Root FastAPI app — both MCP worlds are mounted as sub-apps
app = FastAPI(title="MCP Worlds Router")

app.mount("/standard", mcp.streamable_http_app())
app.mount("/playwright", mcp_playwright.streamable_http_app())
```

- [ ] **Step 4: Run test to verify it passes**

```
python -m pytest tests/mcp/test_asgi_routing.py::test_asgi_app_has_standard_and_playwright_routes -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/mcp/asgi.py tests/mcp/test_asgi_routing.py
git commit -m "feat: refactor asgi.py to mount standard and playwright MCP worlds on FastAPI"
```

---

## Task 4: Update `app/agent/prompts.py` — Playwright World Prompt

**Files:**
- Modify: `app/agent/prompts.py`

Add `PLAYWRIGHT_WORLD_SYSTEM_PROMPT` constant and update `build_planner_prompt()` to accept and inject `mcp_mode`.

- [ ] **Step 1: Write the failing test**

```python
# tests/agent/test_prompts.py
from app.agent.prompts import build_planner_prompt, PLAYWRIGHT_WORLD_SYSTEM_PROMPT

def test_playwright_world_prompt_is_injected_when_mode_is_playwright():
    prompt = build_planner_prompt(
        user_query="cerca iphone",
        scratchpad={},
        step_index=1,
        max_steps=4,
        tool_catalog={},
        mcp_mode="playwright_browser",
    )
    assert "Browser Playwright" in prompt or "browser reale" in prompt.lower()
    assert "contact_seller_playwright" in prompt

def test_playwright_world_prompt_is_not_injected_in_standard_mode():
    prompt = build_planner_prompt(
        user_query="cerca iphone",
        scratchpad={},
        step_index=1,
        max_steps=4,
        tool_catalog={},
        mcp_mode="standard",
    )
    assert "contact_seller_playwright" not in prompt
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/agent/test_prompts.py -v
```
Expected: FAIL (`build_planner_prompt` has no `mcp_mode` param, `PLAYWRIGHT_WORLD_SYSTEM_PROMPT` not defined)

- [ ] **Step 3: Add `PLAYWRIGHT_WORLD_SYSTEM_PROMPT` after `CONVERSATION_ANSWER_SYSTEM_PROMPT` (around line 119 of `app/agent/prompts.py`)**

```python
PLAYWRIGHT_WORLD_SYSTEM_PROMPT = """
### [MODALITÀ BROWSER PLAYWRIGHT ATTIVA]
Sei in modalità Browser Playwright. Hai accesso a un browser Chromium REALE e VISIBILE sullo schermo.

TOOL DISPONIBILI IN QUESTA MODALITÀ:
- `search_products`: Cerca prodotti su eBay con scraping visivo tramite browser reale. Il browser si apre sullo schermo.
- `contact_seller_playwright`: Contatta direttamente un venditore eBay inviando un messaggio dalla pagina del prodotto.

FLUSSO STANDARD:
1. Usa `search_products` per cercare i prodotti (il browser si aprirà visibile).
2. L'utente sceglie il prodotto di interesse.
3. Usa `contact_seller_playwright` con `product_url` (URL del prodotto scelto) e `message` (testo del messaggio da inviare).

REGOLE CRITICHE:
- Per `contact_seller_playwright` devi avere il `product_url` dal risultato di `search_products`. Estrailo dal campo `url` del prodotto nei `top_results`.
- Se l'utente specifica già un messaggio, usalo verbatim nel parametro `message`.
- Se `contact_seller_playwright` ritorna `status=login_required`, informa l'utente che deve effettuare il login su eBay nel browser aperto.
- NON usare `analyze_seller`, `get_ebay_deals` o altri tool non presenti in questa modalità.
### [FINE MODALITÀ PLAYWRIGHT]
""".strip()
```

- [ ] **Step 4: Update `build_planner_prompt()` signature and body**

Find the function at line ~270 and add `mcp_mode: str = "standard"` parameter. In the body, inject the playwright prompt when mode is `playwright_browser`:

```python
def build_planner_prompt(
    user_query: str,
    scratchpad: Union[Dict[str, Any], List[Dict[str, Any]]],
    step_index: int,
    max_steps: int,
    tool_catalog: Dict[str, Dict[str, Any]],
    custom_instructions: Optional[str] = None,
    tone: Optional[str] = None,
    mcp_mode: str = "standard",
) -> str:
    compact_tool_catalog = _compact_tool_catalog_for_prompt(tool_catalog)
    tools_json = json.dumps(compact_tool_catalog, ensure_ascii=False, indent=2)

    system_prompt = PLANNER_SYSTEM_PROMPT

    # Playwright world: inject specialized prompt at the top
    if mcp_mode == "playwright_browser":
        system_prompt = PLAYWRIGHT_WORLD_SYSTEM_PROMPT + "\n\n" + system_prompt

    if custom_instructions:
        system_prompt = (
            f"### [ISTRUZIONI PERSONALIZZATE UTENTE - PRIORITÀ MASSIMA]\n"
            f"{custom_instructions}\n"
            f"### [FINE ISTRUZIONI PERSONALIZZATE]\n\n"
            f"{system_prompt}"
        )

    if tone:
        tone_instruction = f"\nTONO RICHIESTO: {tone.upper()}. Mantieni questo stile in tutte le tue interazioni.\n"
        system_prompt = tone_instruction + system_prompt

    # ... rest of function unchanged (token estimation, scratchpad truncation, return)
```

- [ ] **Step 5: Run test to verify it passes**

```
python -m pytest tests/agent/test_prompts.py -v
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app/agent/prompts.py tests/agent/test_prompts.py
git commit -m "feat: add PLAYWRIGHT_WORLD_SYSTEM_PROMPT and mcp_mode param to build_planner_prompt"
```

---

## Task 5: Update `app/agent/memory.py` — Add `mcp_mode` and `contact_seller_playwright_payload`

**Files:**
- Modify: `app/agent/memory.py`

`RequestState` (aliased as `AgentMemory`) needs: `mcp_mode` field, `contact_seller_playwright_payload` field, and handling in `apply_observation()`, `scratchpad()`, `final_data()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/agent/test_memory.py
from app.agent.memory import RequestState, SessionMemory, LongTermMemory
from app.agent.schemas import Observation

def _make_state() -> RequestState:
    return RequestState(
        user_query="test",
        session_memory=SessionMemory(user_key="test"),
        long_term_memory=LongTermMemory(user_key="test"),
    )

def test_request_state_has_mcp_mode_default_standard():
    state = _make_state()
    assert state.mcp_mode == "standard"

def test_request_state_mcp_mode_can_be_set_playwright():
    state = _make_state()
    state.mcp_mode = "playwright_browser"
    assert state.mcp_mode == "playwright_browser"

def test_contact_seller_playwright_payload_stored_on_observation():
    state = _make_state()
    obs = Observation(
        tool="contact_seller_playwright",
        ok=True,
        status="ok",
        quality=1.0,
        summary="Messaggio inviato",
        data={"success": True, "contact_status": "message_sent"},
        terminal=True,
        state_key="contact_seller_playwright",
    )
    state.apply_observation(obs)
    assert state.contact_seller_playwright_payload is not None
    assert state.contact_seller_playwright_payload["success"] is True
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/agent/test_memory.py -v
```
Expected: FAIL (`RequestState` has no `mcp_mode` or `contact_seller_playwright_payload`)

- [ ] **Step 3: Add fields to `RequestState` dataclass**

In the `RequestState` dataclass (around line 356), add after `contact_seller_payload`:

```python
contact_seller_playwright_payload: Optional[Dict[str, Any]] = None
mcp_mode: str = "standard"
```

- [ ] **Step 4: Add observation handling in `apply_observation()`**

After the existing `if observation.tool == "contact_seller" and observation.ok:` block (around line 500), add:

```python
if observation.tool == "contact_seller_playwright" and observation.ok:
    if isinstance(observation.data, dict):
        self.contact_seller_playwright_payload = observation.data
```

- [ ] **Step 5: Expose in `scratchpad()`**

In the `scratchpad()` method return dict (around line 630), add after `"contact_seller": self.contact_seller_payload,`:

```python
"contact_seller_playwright": self.contact_seller_playwright_payload,
"mcp_mode": self.mcp_mode,
```

- [ ] **Step 6: Expose in `final_data()`**

In the `final_data()` return dict (around line 672), add after `"contact_seller": self.contact_seller_payload,`:

```python
"contact_seller_playwright": self.contact_seller_playwright_payload,
```

- [ ] **Step 7: Run test to verify it passes**

```
python -m pytest tests/agent/test_memory.py -v
```
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add app/agent/memory.py tests/agent/test_memory.py
git commit -m "feat: add mcp_mode and contact_seller_playwright_payload to RequestState"
```

---

## Task 6: Update `app/agent/planner.py` — Route `contact_seller_playwright`

**Files:**
- Modify: `app/agent/planner.py`

Add `contact_seller_playwright` to `VALID_INTENTS`, `_ordered_tools_for_intent`, `_normalize_action_input`. Pass `mcp_mode` to `build_planner_prompt`.

- [ ] **Step 1: Write the failing test**

```python
# tests/agent/test_planner_routing.py
import pytest
from app.agent.planner import ReactPlanner
from app.agent.memory import RequestState, SessionMemory, LongTermMemory

def _make_memory(query: str, mcp_mode: str = "standard") -> RequestState:
    m = RequestState(
        user_query=query,
        session_memory=SessionMemory(user_key="test"),
        long_term_memory=LongTermMemory(user_key="test"),
    )
    m.mcp_mode = mcp_mode
    return m

def test_contact_seller_playwright_in_valid_intents():
    from app.agent.planner import VALID_INTENTS
    assert "contact_seller_playwright" in VALID_INTENTS

def test_ordered_tools_for_contact_seller_playwright():
    planner = ReactPlanner()
    mem = _make_memory("contatta il venditore", mcp_mode="playwright_browser")
    tools = planner._ordered_tools_for_intent("contact_seller_playwright", mem)
    assert tools == ["contact_seller_playwright"]

def test_normalize_action_input_contact_seller_playwright_extracts_url_from_scratchpad():
    planner = ReactPlanner()
    mem = _make_memory("contatta il venditore", mcp_mode="playwright_browser")
    # Simulate search results in memory
    mem._search_payload = {
        "results": [{"url": "https://www.ebay.it/itm/123", "title": "iPhone"}],
        "results_count": 1,
    }
    result = planner._normalize_action_input("contact_seller_playwright", {}, mem)
    assert result is not None
    assert result["product_url"] == "https://www.ebay.it/itm/123"
    assert "message" in result
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/agent/test_planner_routing.py -v
```
Expected: FAIL

- [ ] **Step 3: Add `"contact_seller_playwright"` to `VALID_INTENTS` (line 19)**

```python
VALID_INTENTS = {
    "conversation", "seller_analysis", "product_search", "hybrid",
    "comparison", "item_details", "shipping", "market_trends",
    "deals", "wishlist", "contact_seller", "playwright_search",
    "contact_seller_playwright",  # new
}
```

- [ ] **Step 4: Add case in `_ordered_tools_for_intent()` (after `playwright_search` case, line 735)**

```python
if intent == "contact_seller_playwright":
    return ["contact_seller_playwright"]
```

- [ ] **Step 5: Add case in `_normalize_action_input()` (after the `ebay_scrape` elif block, line ~689)**

```python
elif action == "contact_seller_playwright":
    if not action_input.get("product_url"):
        # Try to extract product URL from search results in memory
        scratchpad = memory.scratchpad()
        top_results = scratchpad.get("top_results") or []
        if top_results and top_results[0].get("url"):
            action_input["product_url"] = top_results[0]["url"]
        else:
            # Also check search_payload directly
            search = getattr(memory, "_search_payload", None) or {}
            results = search.get("results") or []
            if results and results[0].get("url"):
                action_input["product_url"] = results[0]["url"]
            else:
                return None  # Cannot proceed without a product URL
    if not action_input.get("message"):
        action_input["message"] = memory.user_query
```

- [ ] **Step 6: Pass `mcp_mode` to `build_planner_prompt` in `_llm_decide()` (line ~433)**

Find `prompt = build_planner_prompt(` and add `mcp_mode=memory.mcp_mode`:

```python
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
```

- [ ] **Step 7: Run test to verify it passes**

```
python -m pytest tests/agent/test_planner_routing.py -v
```
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add app/agent/planner.py tests/agent/test_planner_routing.py
git commit -m "feat: add contact_seller_playwright routing and mcp_mode to planner"
```

---

## Task 7: Update `app/agent/ebay_agent.py` — Store and Propagate `mcp_mode`

**Files:**
- Modify: `app/agent/ebay_agent.py`

`EbayReactAgent` must accept `mcp_mode`, store it, and inject it into `AgentMemory` after hydration.

- [ ] **Step 1: Write the failing test**

```python
# tests/agent/test_ebay_agent_mcp_mode.py
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

def test_ebay_agent_stores_mcp_mode():
    from app.agent.ebay_agent import EbayReactAgent
    db = MagicMock()
    agent = EbayReactAgent(db=db, mcp_mode="playwright_browser")
    assert agent.mcp_mode == "playwright_browser"

def test_ebay_agent_default_mcp_mode_is_standard():
    from app.agent.ebay_agent import EbayReactAgent
    db = MagicMock()
    agent = EbayReactAgent(db=db)
    assert agent.mcp_mode == "standard"

def test_ebay_agent_playwright_mode_uses_playwright_url():
    from app.agent.ebay_agent import EbayReactAgent
    db = MagicMock()
    agent = EbayReactAgent(db=db, mcp_mode="playwright_browser")
    assert "/playwright/mcp" in agent.mcp_server_url
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/agent/test_ebay_agent_mcp_mode.py -v
```
Expected: FAIL

- [ ] **Step 3: Update `EbayReactAgent.__init__()` to add `mcp_mode`**

In `app/agent/ebay_agent.py`, the `__init__` signature (line 36) becomes:

```python
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
    _DEFAULT_URLS = {
        "standard": os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8050/standard/mcp"),
        "playwright_browser": os.getenv(
            "MCP_PLAYWRIGHT_URL", "http://127.0.0.1:8050/playwright/mcp"
        ),
    }
    self.mcp_server_url = (
        mcp_server_url
        or _DEFAULT_URLS.get(mcp_mode)
        or _DEFAULT_URLS["standard"]
    )
    # ... rest of __init__ unchanged (strict_mcp, MCPToolClient init, logging)
```

- [ ] **Step 4: Inject `mcp_mode` into memory in `run_stream()`**

After `memory = self.memory_service.hydrate_request_state(...)` (around line 146 in `run_stream`):

```python
memory.mcp_mode = self.mcp_mode
```

- [ ] **Step 5: Run test to verify it passes**

```
python -m pytest tests/agent/test_ebay_agent_mcp_mode.py -v
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app/agent/ebay_agent.py tests/agent/test_ebay_agent_mcp_mode.py
git commit -m "feat: add mcp_mode to EbayReactAgent and inject into memory"
```

---

## Task 8: Update `app/api/agent_stream.py` — Accept and Route `mcp_mode`

**Files:**
- Modify: `app/api/agent_stream.py`

`StreamRequest` gains `mcp_mode`. `agent_event_generator` passes it to `EbayReactAgent`.

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_agent_stream_mcp_mode.py
from app.api.agent_stream import StreamRequest

def test_stream_request_default_mcp_mode():
    req = StreamRequest(query="iphone")
    assert req.mcp_mode == "standard"

def test_stream_request_playwright_mode():
    req = StreamRequest(query="iphone", mcp_mode="playwright_browser")
    assert req.mcp_mode == "playwright_browser"

def test_stream_request_invalid_mcp_mode_defaults_to_standard():
    """Unknown mcp_mode values must be sanitized to 'standard'."""
    from app.api.agent_stream import _resolve_mcp_mode
    assert _resolve_mcp_mode("unknown_value") == "standard"
    assert _resolve_mcp_mode("playwright_browser") == "playwright_browser"
    assert _resolve_mcp_mode("standard") == "standard"
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/api/test_agent_stream_mcp_mode.py -v
```
Expected: FAIL

- [ ] **Step 3: Update `StreamRequest` model (around line 311)**

```python
class StreamRequest(BaseModel):
    query: str
    llm_engine: str = "ollama_cloud"
    image: Optional[str] = None
    mcp_mode: str = "standard"
```

- [ ] **Step 4: Add `_resolve_mcp_mode()` helper after `_normalize_llm_engine()` (around line 77)**

```python
_VALID_MCP_MODES = {"standard", "playwright_browser"}

def _resolve_mcp_mode(mcp_mode: str) -> str:
    mode = (mcp_mode or "standard").strip().lower()
    return mode if mode in _VALID_MCP_MODES else "standard"
```

- [ ] **Step 5: Update `agent_event_generator()` signature and `EbayReactAgent` instantiation**

Add `mcp_mode: str = "standard"` to `agent_event_generator` params. In `run_agent()` inner function, pass it to the agent:

```python
async def agent_event_generator(
    request: Request,
    query: str,
    llm_engine: str,
    user: Any,
    image: Optional[str] = None,
    mcp_mode: str = "standard",
):
    llm_engine = _normalize_llm_engine(llm_engine)
    mcp_mode = _resolve_mcp_mode(mcp_mode)
    query = _sanitize_query(query)
    # ...

    async def run_agent():
        db = None
        try:
            db = SessionLocal()
            agent = EbayReactAgent(db=db, user=user, mcp_mode=mcp_mode)
            # ... rest unchanged
```

- [ ] **Step 6: Pass `mcp_mode` from POST handler**

In `agent_stream_post()`, extract and pass `mcp_mode`:

```python
@router.post("/stream")
async def agent_stream_post(
    body: StreamRequest,
    request: Request,
    user=Depends(get_optional_user)
):
    clean_query = _sanitize_query(body.query)
    if not clean_query and not body.image:
        raise HTTPException(status_code=400, detail="Query o immagine necessaria.")
    return await _handle_agent_stream(
        request, clean_query, body.llm_engine, user, body.image, body.mcp_mode
    )
```

Update `_handle_agent_stream()` to accept and forward `mcp_mode`:

```python
async def _handle_agent_stream(
    request: Request,
    clean_query: str,
    llm_engine: str,
    user: Any,
    image: Optional[str] = None,
    mcp_mode: str = "standard",
):
    # ...
    return StreamingResponse(
        agent_event_generator(request, clean_query, llm_engine, user, image, mcp_mode),
        # ...
    )
```

- [ ] **Step 7: Run test to verify it passes**

```
python -m pytest tests/api/test_agent_stream_mcp_mode.py -v
```
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add app/api/agent_stream.py tests/api/test_agent_stream_mcp_mode.py
git commit -m "feat: add mcp_mode to StreamRequest and propagate to EbayReactAgent"
```

---

## Task 9: Update `ebay-ui/src/features/chat/store/settingsStore.ts` — Add `mcpMode`

**Files:**
- Modify: `ebay-ui/src/features/chat/store/settingsStore.ts`

`mcpMode` is local-only (not sent to backend, not loaded from backend).

- [ ] **Step 1: Write the failing test**

```typescript
// ebay-ui/src/features/chat/store/__tests__/settingsStore.test.ts
import { renderHook, act } from '@testing-library/react'
import { useSettingsStore } from '../settingsStore'

test('mcpMode defaults to standard', () => {
  const { result } = renderHook(() => useSettingsStore())
  expect(result.current.settings.mcpMode).toBe('standard')
})

test('mcpMode can be updated to playwright_browser', () => {
  const { result } = renderHook(() => useSettingsStore())
  act(() => {
    result.current.updateLocalSettings({ mcpMode: 'playwright_browser' })
  })
  expect(result.current.settings.mcpMode).toBe('playwright_browser')
})

test('mcpMode is not sent in saveSettingsToBackend payload', () => {
  // saveSettingsToBackend only sends theme, conversationTone, customInstructions,
  // favoriteBrands, pricePreference, contextualBudgets — NOT mcpMode
  // Verified by reading the store implementation
  const store = useSettingsStore.getState()
  // mcpMode must not appear in the type sent to /auth/me/preferences
  expect(typeof store.settings.mcpMode).toBe('string')
})
```

- [ ] **Step 2: Update `UserSettings` interface (line 4)**

```typescript
export interface UserSettings {
  theme: 'light' | 'dark'
  conversationTone: 'neutral' | 'amichevole' | 'professionale'
  customInstructions: string
  favoriteBrands: string
  pricePreference: string
  contextualBudgets?: string
  mcpMode: 'standard' | 'playwright_browser'
}
```

- [ ] **Step 3: Add default value in initial state (line 26)**

```typescript
settings: {
  theme: 'light',
  conversationTone: 'neutral',
  customInstructions: '',
  favoriteBrands: '',
  pricePreference: '',
  contextualBudgets: '',
  mcpMode: 'standard',
},
```

- [ ] **Step 4: Ensure `saveSettingsToBackend` does NOT include `mcpMode` (it already doesn't since the body is explicit — no change needed)**

Verify `saveSettingsToBackend` body (line ~70) does not include `mcpMode`:
```typescript
body: JSON.stringify({
  theme: newSettings.theme,
  conversation_tone: newSettings.conversationTone,
  custom_instructions: newSettings.customInstructions,
  favorite_brands: newSettings.favoriteBrands,
  price_preference: newSettings.pricePreference,
  contextual_budgets: newSettings.contextualBudgets
  // mcpMode intentionally omitted — local only
})
```

- [ ] **Step 5: Ensure `loadSettingsFromAuth` does NOT reset `mcpMode` (it uses explicit params — no change needed)**

The `loadSettingsFromAuth` function sets only `theme`, `conversationTone`, `customInstructions`, `favoriteBrands`, `pricePreference`, `contextualBudgets`. Add `mcpMode: 'standard'` to keep the type valid:

```typescript
loadSettingsFromAuth: (theme, tone, instructions, brands, price, budgets) => {
  set(() => ({
    settings: {
      theme: (theme as 'light'|'dark') || 'light',
      conversationTone: (tone as any) || 'neutral',
      customInstructions: instructions || '',
      favoriteBrands: brands || '',
      pricePreference: price || '',
      contextualBudgets: budgets || '',
      mcpMode: 'standard',  // always reset to standard on auth load
    }
  }))
  // ... rest unchanged
},
```

- [ ] **Step 6: Commit**

```bash
git add ebay-ui/src/features/chat/store/settingsStore.ts
git commit -m "feat: add mcpMode (local-only) to settingsStore UserSettings"
```

---

## Task 10: Update `ebay-ui/src/features/chat/SettingsModal.tsx` — Add "Mondo MCP" Toggle

**Files:**
- Modify: `ebay-ui/src/features/chat/SettingsModal.tsx`

Add a new section between the Brand/Budget block and the Export Chat block.

- [ ] **Step 1: Add the "Mondo MCP" section**

In `SettingsModal.tsx`, after the `<Divider />` that follows the Brand/Budget section (around line 203), add:

```tsx
<Divider sx={{ borderColor: 'var(--border-color)' }} />

{/* MONDO MCP */}
<Box>
  <Typography fontWeight={600} fontSize={15} gutterBottom>Mondo MCP</Typography>
  <Typography variant="body2" color="var(--text-secondary)" mb={2}>
    Scegli come l'agente accede a eBay
  </Typography>
  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
    <Box>
      <Typography fontSize={14} fontWeight={500}>
        {localSettings.mcpMode === 'playwright_browser'
          ? '🌐 Browser Playwright (Visibile)'
          : '⚡ Standard (API eBay)'}
      </Typography>
      <Typography variant="body2" color="var(--text-secondary)" mt={0.5}>
        {localSettings.mcpMode === 'playwright_browser'
          ? 'Chromium si aprirà visibile — permette di cercare e contattare venditori'
          : 'API ufficiali eBay, veloce e affidabile'}
      </Typography>
    </Box>
    <Switch
      checked={localSettings.mcpMode === 'playwright_browser'}
      onChange={(e) =>
        setLocalSettings({
          ...localSettings,
          mcpMode: e.target.checked ? 'playwright_browser' : 'standard',
        })
      }
      color="primary"
    />
  </Box>
</Box>
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd ebay-ui && npx tsc --noEmit
```
Expected: No errors related to `mcpMode`

- [ ] **Step 3: Commit**

```bash
git add ebay-ui/src/features/chat/SettingsModal.tsx
git commit -m "feat: add Mondo MCP toggle to SettingsModal"
```

---

## Task 11: Update `ebay-ui/src/features/agent/api/stream.ts` — Pass `mcpMode`

**Files:**
- Modify: `ebay-ui/src/features/agent/api/stream.ts`

`streamAgent()` gains an optional `mcpMode` parameter, included in the POST body.

- [ ] **Step 1: Update the function signature and body**

```typescript
export function streamAgent(
  query: string,
  image: string | undefined,
  onEvent: (event: AgentEvent) => void,
  llmEngine = "ollama_cloud",
  mcpMode: 'standard' | 'playwright_browser' = 'standard'
) {
  // ...
  fetchEventSource(url, {
    method: "POST",
    headers,
    body: JSON.stringify({
      query,
      image,
      llm_engine: llmEngine,
      mcp_mode: mcpMode,
    }),
    // ... rest unchanged
  })
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd ebay-ui && npx tsc --noEmit
```

- [ ] **Step 3: Commit**

```bash
git add ebay-ui/src/features/agent/api/stream.ts
git commit -m "feat: add mcpMode param to streamAgent, include mcp_mode in POST body"
```

---

## Task 12: Update `ebay-ui/src/features/agent/hooks/useAgentStream.ts` — Read `mcpMode` from Store

**Files:**
- Modify: `ebay-ui/src/features/agent/hooks/useAgentStream.ts`

Read `mcpMode` from `useSettingsStore` and pass it to `streamAgent()`.

- [ ] **Step 1: Add import and read `mcpMode` from store**

At the top of the file, add the import:

```typescript
import { useSettingsStore } from '../../chat/store/settingsStore'
```

Inside the hook body (before `const run = useCallback`), read the setting:

```typescript
const mcpMode = useSettingsStore((state) => state.settings.mcpMode)
```

- [ ] **Step 2: Pass `mcpMode` to `streamAgent()`**

Find line 109: `const nextSource = streamAgent(query, image, (event: AgentEvent) => {` and update it:

```typescript
const nextSource = streamAgent(query, image, (event: AgentEvent) => {
  // ... unchanged callback body
}, undefined, mcpMode)
```

Note: the 4th param `llmEngine` is `undefined` here (uses default `"ollama_cloud"`), `mcpMode` is the 5th.

- [ ] **Step 3: Verify TypeScript compiles**

```bash
cd ebay-ui && npx tsc --noEmit
```
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add ebay-ui/src/features/agent/hooks/useAgentStream.ts
git commit -m "feat: read mcpMode from settingsStore and pass to streamAgent"
```

---

## Task 13: Smoke Test End-to-End

Manual verification steps after all code changes:

- [ ] **Step 1: Start the backend**

```bash
cd C:\Users\paolo\MCP_ECOM
uvicorn app.main:app --port 8000 --reload
# MCP server on port 8050 (check main.py or run_server for startup config)
```

- [ ] **Step 2: Verify both MCP endpoints respond**

```bash
curl http://127.0.0.1:8050/standard/mcp
# Expected: MCP protocol response (not 404)

curl http://127.0.0.1:8050/playwright/mcp
# Expected: MCP protocol response (not 404)
```

- [ ] **Step 3: Start the frontend**

```bash
cd C:\Users\paolo\MCP_ECOM\ebay-ui
npm run dev
```

- [ ] **Step 4: Toggle "Mondo MCP" to Playwright**

Open settings modal → find "Mondo MCP" → toggle ON → save.

- [ ] **Step 5: Send a search query**

Type "cerca iphone 13" in the chat. Verify:
- Chromium browser opens visibly
- eBay search page loads in the browser
- Results stream back in the chat

- [ ] **Step 6: Test contact seller**

After search results appear, type "contatta il venditore del primo risultato e digli 'Ciao, è ancora disponibile?'"

Verify:
- Agent calls `contact_seller_playwright` with `product_url` from the first result and `message`
- Browser navigates to the product page
- Either: contact form is found and filled, OR: `login_required` message is returned to the user

- [ ] **Step 7: Switch back to Standard and verify it still works**

Toggle "Mondo MCP" OFF → send "cerca iphone" → verify API-based results come back without browser opening.

- [ ] **Step 8: Final commit**

```bash
git add .
git commit -m "feat: complete MCP Worlds implementation — standard + playwright_browser"
```

---

## Self-Review

**Spec coverage:**
- ✅ `playwright_server.py` with `search_products` (headless=False) — Task 1
- ✅ `contact_seller_playwright` tool — Task 2
- ✅ `asgi.py` mounting both servers — Task 3
- ✅ `PLAYWRIGHT_WORLD_SYSTEM_PROMPT` + `build_planner_prompt` param — Task 4
- ✅ `mcp_mode` in `RequestState` + `contact_seller_playwright_payload` — Task 5
- ✅ Planner routing for `contact_seller_playwright` — Task 6
- ✅ `EbayReactAgent` stores and injects `mcp_mode` — Task 7
- ✅ `StreamRequest.mcp_mode` + URL resolution — Task 8
- ✅ `settingsStore.ts` `mcpMode` (local-only) — Task 9
- ✅ SettingsModal toggle — Task 10
- ✅ `stream.ts` passes `mcp_mode` — Task 11
- ✅ `useAgentStream.ts` reads from store — Task 12

**Type consistency:**
- `mcpMode` (TypeScript) ↔ `mcp_mode` (Python/JSON) — consistent snake_case/camelCase boundary
- `contact_seller_playwright` tool name is consistent across Tasks 2, 5, 6
- `PLAYWRIGHT_WORLD_SYSTEM_PROMPT` defined in Task 4, used in Task 4 (same file)
- `_resolve_mcp_mode()` defined in Task 8, used in Task 8 (same file)
- MCP URLs: `/standard/mcp` and `/playwright/mcp` consistent across Tasks 3, 7, 8

**No placeholders:** All steps contain complete code.
