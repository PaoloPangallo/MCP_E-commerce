# MCP Architecture Diagrams

UML diagrams for `app/mcp/` — MCP Dual-World Architecture.

## Files

| File | Description |
|------|-------------|
| `mcp_architecture_diagrams.html` | **Interactive HTML viewer** — renders all Mermaid diagrams in-browser |
| `01_dual_world_architecture.md` | Sprint 1: Component diagram — FastAPI mounts two MCP worlds |
| `02_mcp_tools_taxonomy.md` | Sprint 2: Class diagram — tool categories and inventory |
| `03_mcp_client_sequence.md` | Sprint 3: Sequence — tool invocation via MCPToolClient |
| `04_playwright_browser_flow.md` | Sprint 4: Sequence — autonomous Chrome contact flow |
| `05_mcp_data_flow.md` | Sprint 5: Flowchart — resource, profile, wishlist data flow |
| `06_mcp_sync_between_worlds.md` | Sprint 6: Sequence — sync tools/prompts/resources standard → playwright |

## View Diagrams

Open `mcp_architecture_diagrams.html` in a browser (requires internet for Mermaid.js CDN, or use the local `../mermaid.min.js` if present).

Alternatively, view the `.md` files directly — they contain raw Mermaid source that renders in GitHub, GitLab, JetBrains, and any Mermaid viewer.

## Architecture Overview

```
FastAPI (asgi.py)
├── /standard  →  mcp-ecommerce-agent    (API tools: search, seller, item, user)
└── /playwright → mcp-playwright-world   (browser tools: playwright_contact, browser search)
                                       ↕ sync via sync_standard_tools()
```
