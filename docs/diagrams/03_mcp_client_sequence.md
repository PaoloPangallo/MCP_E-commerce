# Sprint 3 — Sequence: Tool Invocation Flow

```mermaid
sequenceDiagram
    participant Agent
    participant Client as "MCPToolClient\n(client.py)"
    participant FastAPI as "FastAPI\n(asgi.py)"
    participant MCPServer as "MCP Server\n(server.py)"
    participant Tool as "MCP Tool\n(tools/*.py)"
    participant DB as "Database\n(SessionLocal)"

    Agent->>Client: call_tool_async("search_products", args)

    alt Local Mode (127.0.0.1/localhost)
        Client->>Client: _get_local_mcp_instance()
        Client->>MCPServer: mcp.call_tool(name, args)
        MCPServer->>Tool: search_products()
        Tool->>DB: _db_context()
        DB-->>Tool: db session
        Tool->>Tool: run_search_pipeline()
        Note over Tool: 1. parse_query_service<br/>2. eBay API search<br/>3. Qdrant RAG search<br/>4. LTR reranking<br/>5. _normalize_search_output()
        Tool-->>MCPServer: normalized dict
        MCPServer-->>Client: CallToolResult(content=[TextContent])
    else HTTP Mode (remote)
        Client->>FastAPI: POST /standard or /playwright
        FastAPI->>MCPServer: route to mounted MCP ASGI app
        MCPServer->>Tool: tool execution
    end

    Client->>Client: _extract_content_parts(raw)
    Note over Client: Handles: CallToolResult, list[TextContent],<br/>list[dict], list[str], str, dict

    Client->>Client: _parse_parts_to_dict(parts)
    Note over Client: 1. json.loads()<br/>2. ast.literal_eval() (Python dicts)<br/>3. raw string fallback

    Client-->>Agent: {status, result, _backend}
```

## Key Classes

### MCPToolClient (client.py)
| Method | Purpose |
|--------|---------|
| `call_tool_async()` | Main entry point for tool invocation |
| `list_tools_async()` | Fetch tool catalog from MCP server |
| `get_tool_schemas_async()` | Fetch inputSchema for all tools |
| `read_resource_async()` | Read MCP resource by URI |
| `get_prompt_async()` | Get prompt by name |

### Mode Detection
- **Local**: `server_url` contains `127.0.0.1` or `localhost` → calls `mcp.call_tool()` directly
- **HTTP**: remote URL → uses `streamable_http_client` + `ClientSession`

## DB Context Flow
```
_tool call → _db_context() → _get_db() → DB session → tool logic → _close_db()
```
- Uses context manager to guarantee session cleanup
- `SessionLocal` factory passed at `configure_mcp()` time
