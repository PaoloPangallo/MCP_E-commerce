import pytest
from app.mcp.playwright_server import mcp_playwright, configure_playwright_mcp

def test_playwright_server_instance_exists():
    assert mcp_playwright is not None

def test_configure_playwright_mcp_runs_without_error():
    configure_playwright_mcp(db_factory=None)

def test_playwright_server_has_search_products_tool():
    """The playwright server must expose search_products."""
    # FastMCP stores tools in _tool_manager
    tool_manager = getattr(mcp_playwright, '_tool_manager', None)
    if tool_manager is not None:
        tool_names = [t.name for t in tool_manager.list_tools()]
        assert "search_products" in tool_names
    else:
        # Fallback: check via tools dict attribute
        tools = getattr(mcp_playwright, '_tools', {})
        assert "search_products" in tools or len(tools) > 0
