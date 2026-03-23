import asyncio
from app.mcp.server import mcp

async def inspect():
    tools = await mcp.list_tools()
    if tools:
        t = tools[0]
        print(f"Tool: {t.name}")
        # MCP standard is 'inputSchema'
        print(f"inputSchema: {getattr(t, 'inputSchema', 'N/A')}")
        # Some wrappers use 'parameters' or 'schema'
        print(f"parameters: {getattr(t, 'parameters', 'N/A')}")
        print(f"schema: {getattr(t, 'schema', 'N/A')}")
        
if __name__ == "__main__":
    asyncio.run(inspect())
