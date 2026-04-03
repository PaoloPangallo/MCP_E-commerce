# Sprint 1 — Component Diagram: Dual-World Architecture

```mermaid
graph TB
    subgraph "FastAPI Host (asgi.py)"
        FA["FastAPI App"]
    end

    subgraph "MCP Standard World (/standard)"
        MCPStd["FastMCP: mcp-ecommerce-agent"]
        ToolsStd["Tools"]
        ResStd["Resources"]
        PromptStd["Prompts"]
    end

    subgraph "MCP Playwright World (/playwright)"
        MCPPW["FastMCP: mcp-playwright-world"]
        ToolsPW["Tools"]
        ToolsPWStd["Synced Tools from Standard"]
        ResPW["Resources"]
        PromptPW["Prompts"]
    end

    subgraph "Tool Registrations"
        SEARCH["search_products"]
        SELLER["analyze_seller"]
        ITEM["get_item_details"]
        PROFILE["update_user_preferences"]
        WISHLIST["manage_wishlist"]
        CONTACT["contact_seller"]
        PWC["contact_seller_playwright"]
        PWS["search_products (browser)"]
    end

    FA -->|/standard| MCPStd
    FA -->|/playwright| MCPPW

    MCPStd --> ToolsStd
    MCPStd --> ResStd
    MCPStd --> PromptStd

    MCPPW --> ToolsPW
    MCPPW --> ToolsPWStd
    MCPPW --> ResPW
    MCPPW --> PromptPW

    ToolsStd --> SEARCH
    ToolsStd --> SELLER
    ToolsStd --> ITEM
    ToolsStd --> PROFILE
    ToolsStd --> WISHLIST
    ToolsStd --> CONTACT

    ToolsPW --> PWC
    ToolsPW --> PWS
    ToolsPWStd -.->|sync via sync_standard_tools()| ToolsStd
```

## Key Points

- **FastAPI** mounts two independent ASGI apps at `/standard` and `/playwright`
- **Standard World** (`mcp-ecommerce-agent`): API-based tools, no browser
- **Playwright World** (`mcp-playwright-world`): Browser automation tools + synced entities
- **sync_standard_tools()** copies tools/prompts/resources from standard → playwright at startup
- Both worlds share the same database factory (`SessionLocal`) via `MCPDependencies`

## File Mapping

| Component | File |
|-----------|------|
| Dual-world mount | `app/mcp/asgi.py` |
| Standard MCP | `app/mcp/server.py` + `app/mcp/core.py` |
| Playwright MCP | `app/mcp/playwright_server.py` |
| Tool registration | `app/mcp/tools/*.py` |
| Resources | `app/mcp/resources.py` |
