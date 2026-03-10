from app.mcp.client import MCPToolClient

URLS = [
    "http://127.0.0.1:8050/mcp",
    "http://127.0.0.1:8050/mcp/",
]

for url in URLS:
    print(f"\n=== TEST {url} ===")
    try:
        client = MCPToolClient(url, enabled=True)
        print("TOOLS:")
        print(client.list_tools())
        print("PROFILE:")
        print(client.call_tool("profile_query", {"query": "iphone 13 massimo 700 euro"}))
    except Exception as e:
        print("FAIL")
        print(repr(e))