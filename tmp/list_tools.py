import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.mcp.server import mcp as standard_mcp
from app.mcp.playwright_server import mcp_playwright

print("--- STANDARD MCP TOOLS ---")
for t in standard_mcp._tool_manager._tools.keys():
    print(f"- {t}")

from app.mcp.playwright_server import sync_standard_tools
# Try syncing tools
sync_standard_tools()

print("\n--- PLAYWRIGHT MCP TOOLS ---")
for t in mcp_playwright._tool_manager._tools.keys():
    print(f"- {t}")

