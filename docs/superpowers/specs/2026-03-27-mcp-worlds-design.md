# MCP Worlds — Design Spec
**Date:** 2026-03-27
**Status:** Approved

---

## Overview

Introduce a "MCP World" selector that lets the user choose between two distinct MCP server environments:

- **Standard World** — existing API-based eBay tools (fast, reliable, uses official eBay API)
- **Playwright Browser World** — browser automation tools using Chromium (visible), enabling scraping and direct UI interactions like contacting sellers

The world is selected from the frontend settings modal and propagated to the backend via the agent stream request.

---

## Architecture

### Dual MCP Server (Backend)

`app/mcp/asgi.py` becomes a FastAPI application that mounts two independent FastMCP servers:

```
FastAPI app (port 8050)
├── /standard/mcp   ← Standard world: all existing API-based eBay tools
└── /playwright/mcp ← Playwright world: search_products (headed) + contact_seller_playwright
```

**`app/mcp/playwright_server.py`** — new file with a dedicated `FastMCP("mcp-playwright-world")` instance. Registers exactly two tools:

1. **`search_products`** — calls `scrape_ebay_search(..., headless=False)`, returns the same normalized structure as the standard `search_products` tool for agent compatibility
2. **`contact_seller_playwright`** — takes `product_url: str` + `message: str`, opens visible Chromium, navigates to the product page, discovers and clicks the "Contatta il venditore" button, fills and submits the contact form

**`app/mcp/tools/playwright_contact.py`** — implementation of the `contact_seller_playwright` tool. Uses the same `_PLAYWRIGHT_EXECUTOR` thread pool and `_run_in_proactor_loop` pattern from `ebay_playwright.py` for Windows compatibility (ProactorEventLoop isolation from uvicorn's SelectorEventLoop).

### Agent Route

**`app/api/agent_stream.py`**

`StreamRequest` gains an optional field:
```python
mcp_mode: str = "standard"  # "standard" | "playwright_browser"
```

URL resolution in `agent_event_generator`:
```
"standard"           → http://127.0.0.1:8050/standard/mcp
"playwright_browser" → http://127.0.0.1:8050/playwright/mcp
```

`EbayReactAgent` already accepts `mcp_server_url` — no change to its constructor needed.

### Planner

**`app/agent/planner.py`**

When `mcp_mode=playwright_browser`, a `PLAYWRIGHT_WORLD_SYSTEM_PROMPT` is prepended to the system prompt injected into the LLM. This tells the agent:
- It has access to a real visible browser
- It can search with visual scraping via `search_products`
- It can contact sellers directly via `contact_seller_playwright` once a product is selected
- The expected flow: search → user picks product → agent calls `contact_seller_playwright` with the product URL and the user's message

### Memory

**`app/agent/memory.py` / `AgentMemory`**

`AgentMemory` gains a `mcp_mode: str = "standard"` field, passed at construction from the agent. The planner reads this to:
- Suggest `contact_seller_playwright` as a natural next step when `search_payload` is populated and `mcp_mode=playwright_browser`
- Include mode context in the scratchpad for LLM reasoning

### Prompts

**`app/agent/prompts.py`**

New constant `PLAYWRIGHT_WORLD_SYSTEM_PROMPT`:
```
Sei in modalità Browser Playwright. Hai accesso a un browser Chromium reale e visibile.
Puoi cercare prodotti con scraping visivo (search_products) e contattare direttamente
i venditori eBay (contact_seller_playwright) una volta che l'utente ha scelto un prodotto.
Flusso standard: 1) cerca con search_products, 2) l'utente sceglie il prodotto,
3) usa contact_seller_playwright con l'URL del prodotto e il messaggio da inviare.
```

---

## Frontend

### `settingsStore.ts`

`UserSettings` gains:
```typescript
mcpMode: 'standard' | 'playwright_browser'
```
Default: `'standard'`.

**Not persisted to backend** — local Zustand state only (resets on page refresh). `saveSettingsToBackend` and `loadSettingsFromAuth` do not include `mcpMode`.

### `SettingsModal.tsx`

New "Mondo MCP" section (above Export Chat, below Brand/Budget):

```
┌─────────────────────────────────────────────┐
│ Mondo MCP                          [Switch] │
│ Standard: API eBay, veloce        OFF = std │
│ Browser Playwright: browser reale ON = play │
└─────────────────────────────────────────────┘
```

Toggle is visually separated with Dividers. When ON, shows a warning chip:
> "Il browser Chromium si aprirà visibile sul desktop"

### `stream.ts`

`streamAgent()` gains `mcpMode` parameter (default `'standard'`). POST body includes:
```json
{ "query": "...", "llm_engine": "...", "mcp_mode": "standard" }
```

### `useAgentStream.ts`

Reads `useSettingsStore().settings.mcpMode` and passes it to `streamAgent()`.

---

## Data Flow

```
User toggles "Browser Playwright" in SettingsModal
  → settingsStore.mcpMode = 'playwright_browser'
  → useAgentStream reads mcpMode
  → streamAgent(query, image, onEvent, llmEngine, 'playwright_browser')
  → POST /agent/stream { query, mcp_mode: 'playwright_browser' }
  → agent_event_generator resolves URL → http://127.0.0.1:8050/playwright/mcp
  → EbayReactAgent(mcp_server_url='http://127.0.0.1:8050/playwright/mcp')
  → MCP client connects to playwright server
  → Planner sees tools: [search_products, contact_seller_playwright]
  → PLAYWRIGHT_WORLD_SYSTEM_PROMPT injected
  → Agent runs search → Chromium opens visibly → results returned
  → User picks product → Agent calls contact_seller_playwright(url, message)
  → Playwright navigates product page → finds contact button → sends message
```

---

## Files to Create / Modify

### New Files
| File | Purpose |
|------|---------|
| `app/mcp/playwright_server.py` | New FastMCP instance for Playwright world |
| `app/mcp/tools/playwright_contact.py` | `contact_seller_playwright` tool implementation |

### Modified Files
| File | Change |
|------|--------|
| `app/mcp/asgi.py` | Becomes FastAPI app mounting both MCP servers |
| `app/api/agent_stream.py` | Add `mcp_mode` to `StreamRequest`, resolve MCP URL |
| `app/agent/memory.py` | Add `mcp_mode` field to `AgentMemory` |
| `app/agent/prompts.py` | Add `PLAYWRIGHT_WORLD_SYSTEM_PROMPT` |
| `app/agent/planner.py` | Inject playwright prompt when mode=playwright_browser |
| `ebay-ui/src/features/chat/store/settingsStore.ts` | Add `mcpMode` to `UserSettings` (local only) |
| `ebay-ui/src/features/chat/SettingsModal.tsx` | Add "Mondo MCP" toggle section |
| `ebay-ui/src/features/agent/api/stream.ts` | Add `mcpMode` param to `streamAgent()` |
| `ebay-ui/src/features/agent/hooks/useAgentStream.ts` | Read `mcpMode` from store and pass to `streamAgent()` |

---

## Error Handling

- If `mcp_mode` value is unrecognized, backend defaults to `"standard"` silently
- `contact_seller_playwright` returns structured error if: login required, contact button not found, form submission fails
- Playwright failures bubble up as `tool_result` events with `ok=false` and a human-readable `summary`

---

## Out of Scope

- Persisting `mcpMode` to the user profile in the database
- A generic "free navigation" browser tool (can be added later)
- Additional Playwright tools beyond `search_products` and `contact_seller_playwright`
- eBay login/authentication handling in Playwright (contact seller requires eBay login — this is a known limitation to address in a follow-up)
