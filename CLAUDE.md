# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Start the application
```powershell
.\start_dev.ps1
```
This starts Docker (Redis, Qdrant, PostgreSQL) and runs `uvicorn app.main:app --reload --port 8050`.

Manual equivalent:
```bash
docker-compose up -d
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8050
```

### Run tests
```bash
pytest app/test/
pytest app/test/test_ebay.py                        # single file
pytest app/test/test_ebay.py::test_function_name    # single test
```

### Database migrations
```bash
alembic revision --autogenerate -m "description"
alembic upgrade head
```

## Architecture

### Request Flow
```
POST /agent/stream
  → EbayReactAgent.run_stream()       app/agent/ebay_agent.py
  → ReactPlanner.decide()             app/agent/planner.py
  → ToolExecutor.execute(ToolCall)    app/agent/executor.py
      → Redis cache check             app/db/redis.py
      → HTTP call to MCP server       app/mcp/client.py → app/mcp/server.py
          → Tool logic                app/mcp/tools/
              → eBay API              app/services/ebay.py
              → Search pipeline       app/services/search_pipeline.py
              → RAG (Qdrant+BM25)     app/services/rag/
  → SSE stream to client (real-time events)
```

### MCP Dual-World Architecture
The MCP server (`app/mcp/asgi.py`) mounts two independent worlds:
- `/mcp/standard/mcp` — core tools: `search_products`, `analyze_seller`, `conversation`, `get_item_details`, `wishlist_tool`, etc.
- `/mcp/playwright/mcp` — browser automation tools: `playwright_contact`, `playwright_search`

Tools in `app/mcp/tools/` are registered at startup; the agent fetches the tool catalog at runtime from the live MCP endpoint.

### Agent Loop (ReAct)
The planner in `app/agent/planner.py` classifies intent and selects a tool. The executor in `app/agent/executor.py` calls the MCP server over HTTP (not direct function calls), caches results in Redis, and builds an `Observation`. The agent loops up to 5 steps until a terminal state is detected, then generates a final answer via LLM.

### Hybrid Search Pipeline
`app/services/search_pipeline.py` orchestrates:
1. LLM query parsing (`app/services/parser.py`) — extracts brand, price range, constraints
2. eBay API query (`app/services/ebay.py`)
3. Dense embedding search + BM25 in Qdrant (`app/services/rag/`)
4. LTR reranking with features: semantic similarity, seller trust score, sentiment, price deviation

### LLM Routing
`app/llm/client.py` is the single LLM call entry point. Primary: Ollama Cloud (`minimax-m2.7:cloud` or `gpt-oss:120b`). Fallback: Gemini (`gemini-1.5-flash`). The `llm_engine` parameter can be passed per-request to override.

### Caching Policy
Defined centrally in `app/config/cache.py`. Cache keys format: `{mcp_mode}:{tool_name}:{hash(input)}`. TTL tiers: 2 min (tool executor), 5 min (general), 1 hr (long-lived), 24 hr (user history). Redis has an in-memory dict fallback if unavailable.

### Key Config
All config via `app/config/settings.py` (Pydantic Settings). Required env vars: `DATABASE_URL`, `REDIS_URL`, `QDRANT_URL`, `EBAY_CLIENT_ID/SECRET/USER_TOKEN`, `OLLAMA_API_KEY`, `GEMINI_API_KEY`, `JWT_SECRET_KEY`.

### Windows / Playwright
`app/main.py` sets `WindowsProactorEventLoopPolicy` at startup — required for Playwright to work on Windows. Do not remove.
