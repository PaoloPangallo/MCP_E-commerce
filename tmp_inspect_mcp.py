from app.mcp.server import mcp
import logging

logging.basicConfig(level=logging.INFO)

print(f"MCP Object Type: {type(mcp)}")
print(f"MCP Dir: {dir(mcp)}")

if hasattr(mcp, "_tools"):
    print("mcp has _tools")
elif hasattr(mcp, "tools"):
    print("mcp has tools")
else:
    print("mcp has NO _tools or tools attribute")

# Try to list them
try:
    from mcp.server.fastmcp import FastMCP
    if isinstance(mcp, FastMCP):
        # Inspect internal server
        if hasattr(mcp, "server"):
            print(f"mcp.server has: {dir(mcp.server)}")
except Exception as e:
    print(f"Error inspecting: {e}")
